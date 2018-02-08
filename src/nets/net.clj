
;;; This namespace provides a function NEW-NET which creates MLP based on the number of inputs, the number of layers, the neurons per layer, and the activation function for a given layer.
(ns nets.net
  (:require [nets.activation-functions :as afs]
            [clojure.algo.generic.math-functions :as math]
            [clojure.core.matrix :as matrix]
            [clojure.core.matrix.implementations :as mat-imp]
            [uncomplicate.neanderthal.core :as nc]
            [uncomplicate.neanderthal.native :as nn]
            [nets.utils :as utils])
  (:gen-class))

(mat-imp/set-current-implementation :clatrix)

(defn- fill-matrix
  "Returns a matrix of dimension M rows and N columns populated with values obtained by calling FUNC. FUNC is a function of 0-arity."
  [num-rows num-cols func]
   (let [weights (take (* num-rows  num-cols) (repeatedly func))]
     (partition num-rows weights)))

; Note: this is ideal if the input is normalized
(defn- weight-initializer
  "Returns a function which returns a random value between (-1/sqrt(d), 1/sqrt(d))"
  [d]
  (let [lim (/ 1 (math/sqrt d))]
    (fn [] (+ (- lim) (rand (* 2 lim))))))

(defn matrices
  "Returns a vector matrices such that the i_th element is the matrix
  connecting the i-1th layer to the ith layer."
  [net]
  (:matrices net))

(defn biases
  "Returns a vector of vectors such that the i_th element is the
  bias vector of the i_th (non-input) layer."
  [net]
  (:biases net))

(defn act-fns
  "Returns a vector of keywords such that the i_th element is the
  activation function of the i_th (non-input) layer."
  [net]
  (:act-fns net))

(defn description?
  "Returns true if NET is a descirption of a a net as opposed
  to a concrete implementation."
  [net]
  (map? net))

(defn new-net
  "Returns a new net with NUM-INPUTS inputs. The remaining input are vectors of size two, the 0th value describing the amount of neurons in the layer and the 1th value a keyword corresponding to one of the activation function as specified in nets.activation-functions. Currently each layer is assumed to be fully connected and the weights are chosen to be between (-1/sqrt (d), 1/sqrt (d)) where d is the number of neurons in the preceding layer. In addition a bias vector will be initialized in all layers as the 0 vector.
  For example:
  (new-net 64 [10 :relu] [5 :sigmoid])"
  [num-inputs & layers]
  (let [dimensions (mapv #(first %) layers)
        matrices (mapv #(fill-matrix %1 %2 (weight-initializer %1))
                       (into [num-inputs] (butlast dimensions))
                       dimensions)
        biases (mapv #(matrix/array (repeat % 0)) dimensions)
        act-fns (mapv second layers)]
    {:num-inputs num-inputs
     :matrices matrices
     :biases biases
     :act-fns act-fns}))


; Displaying MLP, does not work ATM
(defn- show-layer
  "Shows the layer-size, the matrix connecting the previous layer to this layer,
  bias vectors, and the name of the associated activation function."
  [layer]
  (matrix/pm (:matrix layer))
  (printf "Bias: ")
  (print (:bias layer))
  (newline)
  (printf "Activation function: %s\n" (:act-fn layer))
  (printf "Layer, size: %d ==================================\n"
          (matrix/dimension-count (:matrix layer) 1)))

(defn show-net
  "A pretty printer for nets"
  [net]
  (printf "Input layer, size: %d =================================\n"
          (:num-inputs net))
  (dorun (map show-layer (:layers net))))

; Serializing
(defn to-string
  "Produces a string representation of the net that can be read to produce a net
  using from-string. Each layer is reduced to a vector of the following form
  [MATRIX BIAS ACT-FUN-NAME]. From-string can be used to process this data into a
  usable net."
  [net]
  (pr-str (mapv vector (matrices net) (biases net) (act-fns net))))

(defn from-string
  "Produces a net from a string produced by to-string."
  [s]
  (let [net-data (clojure.edn/read-string s)
        mats     (map first net-data)
        bs       (map second net-data)
        afs (map #(nth % 2) net-data)]
    {:num-inputs (matrix/column-count (first mats))
     :layers
     (mapv
      (fn [mat bias af-name]
        {:matrix mat :bias bias :act-fn af-name})
      mats bs afs)}))


(defn to-file
  "Saves NET to file named FNAME."
  [net fname]
  (spit fname (to-string net)))
(defn from-file
  "Read a net from the given file."
  [fname]
  (from-string (slurp fname)))
