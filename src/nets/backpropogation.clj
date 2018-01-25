(ns nets.core
  (:require [clojure.algo.generic.math-functions :as math]
            [clojure.core.matrix :as matrix]
            [clojure.test :as test]
            [nets.net :as net]
            [clojure.math.numeric-tower])
  (:gen-class))

;;; This namespace implements the backpropogation algorithm. It follows the convention
;;; that vectors are row vectors (as opposed to column vectors) so matrix
;;; multiplication is (v * M).
(defmacro in-net
  "ANAPHORIC MACRO, binds unqualified symbols: matrices biases act-fns deriv-fns
  to vectors corresponding to the layer data. For example,
  (nth matrices 0) === (nth (:layers net) 0)  ."
  [net body]
  `(let [~'matrices (mapv #(:matrix %) (:layers ~net))
         ~'biases (mapv #(:bias %) (:layers ~net))
         ~'act-fns (mapv #(:act-fn %) (:layers ~net))
         ~'deriv-fns (mapv #(:deriv-fn %) (:layers ~net))]
     ~body))

; TODO: cache pre-activation values as well b/c those are used in backpropogate
(defn propogate-forward
  "Propogates an input through the net, returning the intermediate layer outputs. The
  return vector has the following structure:
  [input layer-0-output layer-1-output ... output(output-layer-output)]"
  [net input]
  {:pre [(= (:num-inputs net) (count input))]
   :post [(= (count (:layers net)) (dec (count %)))]}
  (in-net net
          (reductions
           (fn [out [matrix bias act-fn]]
           ;Compute the output of the next layer
             (act-fn (matrix/add (matrix/mmul out matrix) bias)))
           input
           (mapv vector matrices biases act-fns))))

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

(defmacro showlet
  "Performs a let binding as usual, but prints out the value of each binding immediately
  after the binding succeeds."
  [letvec body]
  (if (empty? letvec)
    body
    (let [[symbol expr & r] letvec]
      `(let [~symbol ~expr]
         (do (print ~(name symbol) ~symbol)
             (newline)
             (showlet ~r ~body))))))

;; TODO: change fprop so that it caches the right value isntead of recomputing it here
(defn backpropogate
  [net fprop-values true-value deriv-error-fn]
  {:pre [(= (count (last fprop-values)) (count true-value))]
   :post [(= (count (:layers net)) (count %))]}
  (in-net net
          (let [output (last fprop-values)
                error-deriv (deriv-error-fn output true-value)
                ; Compute the preactivation layer-inputs
                layer-inputs (mapv #(matrix/add (matrix/mmul %1 %2) %3)
                                   (butlast fprop-values) matrices biases)
                ; Compute the error-signal at the output layer
                error-signal (hprod error-deriv
                                    ((last deriv-fns) (last layer-inputs)))]
            (reverse
             (reductions
              (fn [prev-error-signal [prev-matrix layer-deriv]]
                (hprod (matrix/mmul prev-error-signal prev-matrix)
                       layer-deriv))
              error-signal
              (reverse
              ; in order to perform backpropogation for the i_th layer
              ; we need the transpose of the i+1th matrix
              ; and the derivation of i_th layer input value
               (mapv vector
                     (mapv #(matrix/transpose %) (rest matrices))
                     (mapv #(%1 (matrix/mmul %2 %3))
                           (butlast deriv-fns)
                           (butlast layer-inputs)
                           (butlast matrices)))))))))

(defn deriv-matrices
  "To calculate the derivative of the error with respect to i_th weight matrix, the equation is
  y_(i - 1) o delta_i where y_j is the transpose of the output of the j_th layer and d_j the j_th's layer delta. In order to use the matrix multiplication, the arguments (which are vectors of vectors) are converted to vectors of matrices."
  [fprop-values bprop-values]
  (let [ys (mapv #(matrix/transpose %) (mapv vector fprop-values))
        ds (mapv vector bprop-values)]
    (mapv #(matrix/mmul %1 %2) ys ds)))

(defn weight-update
  "Produces a new net given the layer-outputs FPROP-VALS, layer deltas BPROP_VALS and a learning rate."
  [net fprop-vals bprop-vals learning-rate]
  (in-net net
          (let [derivs (deriv-matrices fprop-vals bprop-vals)
                scaled-matrices (mapv #(matrix/scale learning-rate %) matrices)
                weight-update-matrices (mapv #(hprodm %1 %2) scaled-matrices derivs)
                new-matrices (mapv #(matrix/sub %1 %2) matrices weight-update-matrices)
                new-biases (mapv #(hprod %1 %2) biases fprop-vals)]
            (assoc net :layers
                   (mapv (fn [layer matrix bias]
                           (assoc layer :matrix matrix :bias bias))
                         (:layers net)
                         new-matrices
                         new-biases)))))
(defn train
  "Given an INPUT, a TARGET output, a LEARNING-RATE, and the derivative
  of the error function, ERROR-DERIV, train will retrun the net
  resulting from performing backpropogation on this particular input."
  [net input target learning-rate error-deriv]
  (let [fvals (propogate-forward net input)
        bvals (backpropogate net fvals target error-deriv)]
    (weight-update net fvals bvals learning-rate)))
