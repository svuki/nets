(ns nets.core
  (:require [clojure.algo.generic.math-functions :as math]
            [clojure.core.matrix :as matrix]
            [clojure.test :as test]
            [nets.net :as net]
            [clojure.math.numeric-tower])
  (:gen-class))

(defmacro in-net
  "ANAPHORIC, binds unqualified symbols: matrices biases act-fns deriv-fns."
  [net body]
  `(let [~'matrices (mapv #(:matrix %) (:layers ~net))
         ~'biases (mapv #(:bias %) (:layers ~net))
         ~'act-fns (mapv #(:act-fn %) (:layers ~net))
         ~'deriv-fns (mapv #(:deriv-fns %) (:layers ~net))]
     ~body))

(defn propogate-forward
  [net input]
  (if (not (= (:num-inputs net) (count input)))
    (printf "Error, net was passed input of size %d but its input layer is of size %d.\n"
            (count input) (:num-inputs net))
    (in-net net
            (reductions
             (fn [out [matrix bias act-fn]]
               ; Compute the output of the next layer
               (act-fn (matrix/add (matrix/mmul out matrix) bias)))
             input
             (mapv vector matrices biases act-fns)))))

(defn- hprod
  "Returns the componentwise product of vectors V1 and V2."
  [v1 v2]
  (mapv * v1 v2))
(defn- hprodm
  "Returns the Hadamard product of matrics M1 and M2"
  [m1 m2]
  (mapv #(hprod %1 %2) m1 m2))

(defn- mapvmapv
  [func vector]
  (mapv #(mapv func %) vector))


;; TODO: change fprop so that it caches the right value isntead of recomputing it here
(defn backpropogate
  [net fprop-values true-value deriv-error-fn];deriv-error-fn takes vec and tval to err-vec
  (in-net net
          (let [output (last fprop-values)
                error-deriv (deriv-error-fn output true-value)
                rev-inputs (reverse
                            (mapv
                            #(%1 (matrix/add (matrix/mmul %2 %3) %4))
                            deriv-fns
                            (butlast fprop-values)
                            matrices
                            biases))
                rev-mats (reverse (mapv #(matrix/transpose %) matrices))
                error-signal (hprod error-deriv (first rev-inputs))]
            (print 'here)
            (reverse ; Given that the error propogates backwards, we reverse the result
             (reductions
              (fn [prev-error-signal [prev-matrix layer-input]]
                (hprod (matrix/mmul prev-error-signal prev-matrix) layer-input)
                error-signal
                (mapv vector rev-mats (rest rev-inputs))))))))
(defn deltas
  "Given an error function ERROR-FN (type: VEC -> VCE -> NUM), the pre-activation outputs OUTPUTS of each layer of the net represented
  by MATRICES, the derivative of the activation function DERIV-ACT-FN (type: VEC -> VEC) used in the forward propogation to obtain OUTPUTS,
  and the target-value of the computation TARGET-VALUE, deltas will perform backwards propogation of the error, returning
  a vector of vectors such that the i_th elements of the j_th vector contains the error associated with the i_th neuron of the j_th layer."
  [matrices net-outputs true-value deriv-act-fn]
  (let [net-result (last net-outputs)
        rev-layer-in (reverse
                      (mapv #(deriv-act-fn (matrix/mmul %1 %2))
                            (butlast net-outputs) matrices))
        rev-trans-mats (reverse (mapv #(matrix/transpose %) (rest matrices)))]
    (reverse
     (reductions
      ; The reduction function backpropgates the deltas from the preceding layer
      (fn [preceding-delta [preceding-matrix-trans layer-input]]
        (hprod (matrix/mmul preceding-delta preceding-matrix-trans) layer-input))
      ; The initial value of the reduction is the delta of the outputs layer
      (hprod (mapv - net-result true-value) (first rev-layer-in))
      ; Subsequent deltas have to be backpropogated through the matrices
      (mapv vector rev-trans-mats (rest rev-layer-in))))))

(defn weight-update
  [matrices deltas layer-outputs learning-rate]
  ;;; TODO: find a nicer way of getting matrices from the vectors...
  (let [deltas-mats (mapv vector deltas)
        outputs-transp-mats (mapv #(matrix/transpose (vector %)) (butlast layer-outputs))
        partials (mapv #(matrix/mmul %1 %2) outputs-transp-mats deltas-mats)]
        ;;; this map is awkard ^^^ why doesn't matrix.core allow matrix/mmul to work with just vectors?
    (mapv #(matrix/sub %1 (hprodm (matrix/scale %1 learning-rate) %2))
          matrices partials)))


(defn train
  "Given INPUT and TARGET-OUTPUT, train updates MATRICES."
  [matrices input target-output learning-rate act-fun deriv-act-fn]
  (let [layer-outs (propogate-forward matrices input act-fun)
        delts (deltas matrices layer-outs target-output deriv-act-fn)]
    (weight-update matrices delts layer-outs learning-rate)))
;; notes: erro-fn and deriv-act-fn need to either apply to vectors or to individual numbers... decide.


(def mats [[[3 2 0] [-1 0 0]] [[5] [-2] [-8]]])
(defn error-fn [v1 v2] (mapv - v1 v2))
(defn  act-fn [v] (mapv #(+ 1 %) v))
(defn deriv-act-fn [v] (mapv #( * 0.5 %) v))
(def outputs (propogate-forward mats [1 1] act-fn))
