(ns nets.backpropogation
  (:require [clojure.algo.generic.math-functions :as math]
            [clojure.core.matrix :as matrix]
            [clojure.test :as test]
            [nets.net :as net]
            [clojure.math.numeric-tower]
            [nets.activation-functions :as afs]
            [nets.error-functions :as efs]
            [uncomplicate.neanderthal.core :as n-core])
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
          (act-fn (matrix/add (matrix/mmul matrix out) bias)))
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
           (let [n-pact (matrix/add (matrix/mmul matrix l-out) bias)
                 ; might need to use emap here
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

; hacky: and if v is a float? or a list?
(defn- hprod-or-mmul
  "Receives two arguments X1 and X2. If both are vectors, then returns the
  component-wise product (hadamard product) of X1 and X2. If X2 is a matrix,
  returns the matrix multiplication X1 and X2. This is done to accomodate activation
  functions like softmax where the whole jacobian must be used to compute the error.
  For now, X1 should never be a matrix."
  [x1 x2]
  {:pre [(matrix/vec? x2)]}
  (if (matrix/matrix? x1)
    (matrix/mmul x1 x2)
    (matrix/emul x1 x2)))

(defn- smax-cent?
  "Returns true if the net has a softmax output layer and the cost function
  is the cross-entropy-function"
  [maybe-smax maybe-cent]
  (and (= maybe-smax (afs/get-deriv :softmax))
       (= maybe-cent (efs/get-cost-grad :cross-entropy))))

(defn- soft-max-cross-entropy-error-signal
  "A simplified formula to handle the use of softmax on the output layer
  with cross entropy for the cost function. This reduces the rate of
  arithmetic exceptions raised when computing the two seperately for this
  computation"
  [actual target]
  (matrix/sub actual target))

; TODO: change this
(defn error-signal
  "Calculates the error-signal at the output layer using the map
  produced by forward-propogate-outer FPROP-MAP, the gradient of the
  cost chosen cost function COST-GRAD-FN, and the target output TARGET."
  [fprop-map target cost-grad-fn deriv-fn]
  (if (smax-cent? deriv-fn cost-grad-fn)
    (soft-max-cross-entropy-error-signal
     (:net-output fprop-map)
     target)
    (let [err-gradient (cost-grad-fn (:net-output fprop-map) target)
          layer-grad (deriv-fn (last (:layer-preact fprop-map)))]
      (hprod-or-mmul layer-grad err-gradient))))

(defn deltas
  [net fprop-map target cost-grad-fn]
  (net/in-net net
      (let [output-delta
            (error-signal fprop-map target cost-grad-fn (last deriv-fns))]
        (reverse
         (reductions
          (fn [delta-i+1 [matrix-t-i+1 deriv-preact-v-i]]
            (hprod-or-mmul
             deriv-preact-v-i
             (matrix/mmul matrix-t-i+1 delta-i+1)))
          output-delta
          (reverse
           (mapv vector
                 (map #(matrix/transpose %) (rest matrices))
                 (map #(%1 %2)
                      (butlast deriv-fns)
                      (butlast (:layer-preact fprop-map))))))))))

(defn deriv-matrices
  "Returns a vector of matrices such that the i,j_th value of the k_th
  matrix is the derivative of the cost function (used to calculate the
  layer deltas) wit respect to the i,h_th weight in the k_th weight matrix."
  [layer-outputs deltas]
  (map matrix/outer-product deltas layer-outputs))

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
        new-biases (mapv #(matrix/sub %1 (matrix/scale lrate %2))
                         (net/biases net) deltas)]
    (assoc net :layers
           (mapv (fn [layer matrix bias]
                   (assoc layer :matrix matrix :bias bias))
                 (:layers net)
                 new-matrices
                 new-biases))))
(defn sgd
  "Implements stochastic gradient descent.
  Given an INPUT, a TARGET output, a LEARNING-RATE, and the derivative
  of the error function, ERROR-DERIV, train will retrun the net
  resulting from performing backpropogation on this particular input."
  [net input target lrate cost-gradient-fn]
  (let [fprop-map (propogate-forward-outer net input)
        ds (deltas net fprop-map target cost-gradient-fn)]
    (weight-update net (:layer-outputs fprop-map) ds lrate)))
