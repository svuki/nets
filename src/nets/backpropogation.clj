(ns nets.backpropogation
  (:require [clojure.algo.generic.math-functions :as math]
            [clojure.core.matrix :as matrix]
            [clojure.test :as test]
            [nets.net :as net]
            [clojure.math.numeric-tower])
  (:gen-class))

;;; This namespace implements the backpropogation algorithm. It follows the convention
;;; that vectors are row vectors (as opposed to column vectors) so matrix
;;; multiplication is (v * M).

; TODO: cache pre-activation values as well b/c those are used in backpropogate
(defn net-eval
  "Passes INPUT through NET retruning the result."
  [net input]
  {:pre [(= (:num-inputs net) (count input))]}
  (net/in-net net
              (reduce
               (fn [out [matrix bias act-fn]]
                 ;Compute the output of the next layer
                 (act-fn (matrix/add (matrix/mmul out matrix) bias)))
               input
               (mapv vector matrices biases act-fns))))

(defn propogate-forward-inner
  "Propogates an INPUT through NET, a vector of the form [V1 V2 X]
  where V1 is a vector of the layer-outputs, V2 a vector of the layer-preactivation
  values and X the outputs of the net. Note that The last element of V1 is the output
  of the last hidden layer."
  [net input]
  (net/in-net net
              (reduce
               (fn [[l-outv l-pactv l-out] [matrix bias act-fn]]
                 (let [n-pact (matrix/add (matrix/mmul l-out matrix) bias)
                       n-out (act-fn n-pact)]
                   [(conj l-outv l-out)
                    (conj l-pactv n-pact)
                    n-out]))
               [[] [] input]
               (mapv vector matrices biases act-fns))))

(defn propogate-forward-outer
  "Takes NET and INPUT and passes them to propogate-forward-inner, converting
   the returned values to the map
  {:layer-outputs V1 :layer-preact V2 :net-output X}
  V1 is a vector whose i_th component is the output of the i_th layer (the 0
  layer is the input layer whose output is simply the input itself).
  V2 is a vector whose i_th component is the preactivation value of
  i + 1 layer (the input value has no preactivation value).
  X is the vector outputted by the whole net"
  [net input]
  (let [fprop-data (propogate-forward-inner net input)]
    (zipmap [:layer-outputs :layer-preact :net-output] fprop-data)))

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
  [net fprop-values true-value deriv-error-fn]
  {:pre [(= (count (last fprop-values)) (count true-value))]
   :post [(= (count (:layers net)) (count %))]}
  (net/in-net net
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
                     (mapv #(%1 %2)
                           (butlast deriv-fns)
                           (butlast layer-inputs)))))))))

(defn deriv-matrices
  "To calculate the derivative of the error with respect to i_th weight matrix, the equation is
  y_(i - 1) o delta_i where y_j is the transpose of the output of the j_th layer and d_j the j_th's layer delta. In order to use the matrix multiplication, the arguments (which are vectors of vectors) are converted to vectors of matrices."
  [fprop-values bprop-values]
  (let [ys (mapv #(matrix/transpose %) (mapv vector fprop-values))
        ds (mapv vector bprop-values)]
    (mapv #(matrix/mmul %1 %2) ys ds)))

(defn weight-update
  "Produces a new net given the layer-outputs FPROP-VALS, layer deltas BPROP-VALS and a LEARNING-RATE."
  [net fprop-vals bprop-vals learning-rate]
  (net/in-net net
          (let [derivs (deriv-matrices fprop-vals bprop-vals)
                scaled-matrices (mapv #(matrix/scale learning-rate %) matrices)
                weight-update-matrices (mapv #(hprodm %1 %2) scaled-matrices derivs)
                new-matrices (mapv #(matrix/sub %1 %2) matrices weight-update-matrices)
                new-biases (mapv #(mapv - %1 (matrix/scale learning-rate %2))
                                 biases bprop-vals)]
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
