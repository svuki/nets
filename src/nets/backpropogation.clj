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

(defn error-signal
  "Calculates the error-signal at the output layer using the map
  produced by forward-propogate-outer FPROP-MAP, the gradient of the
  cost chosen cost function COST-GRAD-FN, and the target output TARGET."
  [fprop-map target cost-grad-fn deriv-fn]
  (let [err-gradient (cost-grad-fn (:net-output fprop-map) target)
        layer-grad (deriv-fn (last (:layer-preact fprop-map)))]
    (hprod err-gradient layer-grad)))

(defn deltas
  [net fprop-map target cost-grad-fn]
  (net/in-net net
      (let [output-delta
            (error-signal fprop-map target cost-grad-fn (last deriv-fns))]
        (reverse
         (reductions
          (fn [delta-i+1 [matrix-t-i+1 deriv-preact-v-i]]
            (hprod (matrix/mmul delta-i+1 matrix-t-i+1)
                   deriv-preact-v-i))
          output-delta
          (reverse
           (mapv vector
                 (mapv #(matrix/transpose %) (rest matrices))
                 (mapv #(%1 %2)
                       (butlast deriv-fns)
                       (butlast (:layer-preact fprop-map))))))))))

(defn deriv-matrices
  "Returns a vector of matrices such that the i,j_th value of the k_th
  matrix is the derivative of the cost function (used to calculate the
  layer deltas) wit respect to the i,h_th weight in the k_th weight matrix."
  [layer-outputs deltas]
  ;;; To use matrix/transpose we need to cast our vectors as single
  ;;; row matrices
  (let [ys (mapv #(matrix/transpose %) (mapv vector layer-outputs))
        ds (mapv vector deltas)]
    (mapv #(matrix/mmul %1 %2) ys ds)))

(defn weight-update-matrices
  "Given a net NET, layer-outputs L-OUTS, a learning-rate LRATE, and layer
  deltas DELTAS, produces a vector of updated matrices such that the i_th
  component is the updated matrix of the i_th layer in the net."
  [net l-outs lrate deltas]
  (let [matrices (net/matrices net)
        derivs (deriv-matrices l-outs deltas)
        scaled-derivs (mapv #(matrix/scale lrate %) derivs)]
    (mapv #(matrix/sub %1 %2) matrices scaled-derivs)))

(defn weight-update
  "Produces a new net with updated matrices according to LRATE."
  [net l-outs deltas lrate]
  (let [new-matrices (weight-update-matrices net l-outs lrate deltas)
        new-biases (mapv #(mapv - %1 (matrix/scale lrate %2))
                         (net/biases net) deltas)]
    (assoc net :layers
           (mapv (fn [layer matrix bias]
                   (assoc layer :matrix matrix :bias bias))
                 (:layers net)
                 new-matrices
                 new-biases))))
(defn train
  "Given an INPUT, a TARGET output, a LEARNING-RATE, and the derivative
  of the error function, ERROR-DERIV, train will retrun the net
  resulting from performing backpropogation on this particular input."
  [net input target lrate cost-gradient-fn]
  (let [fprop-map (propogate-forward-outer net input)
        ds (deltas net fprop-map target cost-gradient-fn)]
    (weight-update net (:layer-outputs fprop-map) ds lrate)))
