
;;; This namespace provides a function NEW-NET which creates MLP based on the number of inputs, the number of layers, the neurons per layer, and the activation function for a given layer.
(ns nets.net
  (:require [nets.activation-functions :as afs]
            [clojure.algo.generic.math-functions :as math]
            [clojure.core.matrix :as matrix]
            [clojure.core.matrix.implementations :as mat-imp]
            [nets.utils :as utils])
  (:gen-class))

(mat-imp/set-current-implementation :clatrix)

(defn- fill-matrix
  "Returns a matrix of dimension M rows and N columns populated with values obtained by calling FUNC. FUNC is a function of 0-arity."
  [num-rows num-cols func]
  (matrix/matrix
   (let [weights (take (* num-rows  num-cols) (repeatedly func))]
     (partition num-rows weights))))

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
  (mapv :matrix (:layers net)))

(defn biases
  "Returns a vector of vectors such that the i_th element is the
  bias vector of the i_th (non-input) layer."
  [net]
  (mapv :bias (:layers net)))

(defn act-fns
  "Returns a vector of functions such that the i_th element is the
  activation function of the i_th (non-input) layer."
  [net]
  (mapv :act-fn (:layers net)))

(defn deriv-fns
  "Returns a vector of functions such that the i_th element is the
  derivative of the i_th activation function."
  [net]
  (mapv :deriv-fn (:layers net)))

(defn af-names
  "Returns a vector of the keywords used to obtain the activation functions. The i_th
  element corresponds to the i_th (non-input) layer."
  [net]
  (:af-names net))

(defmacro in-net
  "ANAPHORIC MACRO, binds unqualified symbols: matrices biases act-fns deriv-fns
  to vectors corresponding to the layer data. For example,
  (nth matrices 0) === (nth (:layers net) 0)."
  [net body]
  `(let [~'matrices (matrices ~net)
         ~'biases (biases ~net)
         ~'act-fns (act-fns ~net)
         ~'deriv-fns (deriv-fns ~net)]
     ~body))

(defn- good-dimensions?
  "Takes a net NET and ensures the dimensions of its matrices fit together.
  Note that the input vectors are assumed to be row vectors, so the matrix's
  column size describes the dimensions of the input."
  [net]
  (let [dims (map matrix/shape (matrices net))
        row-sizes (map first dims)
        col-sizes (map second dims)]
    (and (= (:num-inputs net) (second (first dims)))
         (not-any? false? (map = (rest col-sizes) (butlast row-sizes))))))

(defn new-net
  "Returns a new net with NUM-INPUTS inputs. The remaining input are vectors of size two, the 0th value describing the amount of neurons in the layer and the 1th value a keyword corresponding to one of the activation function as specified in nets.activation-functions. Currently each layer is assumed to be fully connected and the weights are chosen to be between (-1/sqrt (d), 1/sqrt (d)) where d is the number of neurons in the preceding layer. In addition a bias vector will be initialized in all layers as the 0 vector.
  For example:
  (new-net 64 [10 :relu] [5 :sigmoid])"
  [num-inputs & layers]
  {:pre [(not (nil? layers))]
   :post [(not-any? nil? (act-fns %))
          (not-any? nil? (deriv-fns %))
          (good-dimensions? %)]}
  (let [dimensions (mapv #(first %) layers)
        matrices (mapv #(fill-matrix %1 %2 (weight-initializer %1))
                       (into [num-inputs] (butlast dimensions))
                       dimensions)
        biases (mapv #(matrix/array (repeat % 0)) dimensions)
        act-fns (mapv #(afs/get-fn (second %)) layers)
        deriv-fns (mapv #(afs/get-deriv (second %)) layers)]
    {:num-inputs num-inputs
     :layers (mapv (fn [m b a d] {:matrix m :bias b :act-fn a :deriv-fn d})
                   matrices biases act-fns deriv-fns)
     :af-names (mapv #(second %) layers)}))


(defn- show-layer
  "Shows the layer-size, the matrix connecting the previous layer to this layer,
  bias vectors, and the name of the associated activation function."
  [layer act-fn-kword]
  (matrix/pm (:matrix layer))
  (printf "Bias: ")
  (print (:bias layer))
  (newline)
  (printf "Activation function: %s\n" (name act-fn-kword))
  (printf "Layer, size: %d ==================================\n"
          (matrix/dimension-count (:matrix layer) 1)))

(defn show-net
  "A pretty printer for nets"
  [net]
  (printf "Input layer, size: %d =================================\n"
          (:num-inputs net))
  (dorun (map show-layer (:layers net) (:af-names net))))

(defn to-string
  "Produces a string representation of the net that can be read to produce a net
  using from-string. Each layer is reduced to a vector of the following form
  [MATRIX BIAS ACT-FUN-NAME]. From-string can be used to process this data into a
  usable net."
  [net]
  (let [mats (map matrix/to-nested-vectors (matrices net))
        bs   (map vec (biases net))
        afs  (af-names net)]
    (pr-str (mapv vector mats bs afs))))

(defn from-string
  "Produces a net from a string produced by to-string."
  [s]
  (let [net-data (read-string s)
        mats     (map #(-> % first matrix/matrix) net-data)
        bs       (map #(-> % second matrix/array) net-data)
        af-names (map #(last %) net-data)
        afuns    (map afs/get-fn af-names)
        dfuns    (map afs/get-deriv af-names)]
    {:num-inputs (matrix/column-count (first mats))
     :af-names   af-names
     :layers
     (mapv
      (fn [mat bias a-fun d-fun]
        {:matrix mat :bias bias :act-fn a-fun :deriv-fn d-fun})
      mats bs afuns dfuns)}))


(defn to-file
  "Saves NET to file named FNAME."
  [net fname]
  (spit fname (to-string net)))
(defn from-file
  "Read a net from the given file."
  [fname]
  (from-string (slurp fname)))
