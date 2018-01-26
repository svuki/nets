;;; This namespace provides a function NEW-NET which creates MLP based on the number of inputs, the number of layers, the neurons per layer, and the activation function for a given layer.
(ns nets.net
  (:require [nets.activation-functions :as afs]
            [clojure.algo.generic.math-functions :as math]
            [clojure.core.matrix :as matrix]
            [nets.net.utils :as utils])
  (:gen-class))

(defn- fill-matrix
  "Returns a matrix of dimension M rows and N columns populated with values obtained by calling FUNC. FUNC is a function of 0-arity."
  [num-rows num-cols func]
  (let [weights (take (* num-rows  num-cols) (repeatedly func))]
    (partition num-cols weights)))

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
  bias vector of the i_th layer."
  [net]
  (mapv :bias (:layers net)))

(defn act-fns
  "Returns a vector of functions such that the i_th element is the
  activation function of the i_th layer."
  [net]
  (mapv :act-fn (:layers net)))

(defn deriv-fns
  "Returns a vector of functions such that the i_th element is the
  derivative of the i_th activation function."
  [net]
  (mapv :deriv-fn (:layers net)))

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

; TODO: test
(defn- good-dimensions?
  "Takes a net NET and ensures the dimensions of its matrices fit together.
  Note that the input vectors are assumed to be row vectors, so the matrix's
  column size describes the dimensions of the input."
  [net]
  (let [row-size (fn [m] (matrix/dimension-count m 0))
        col-size (fn [m] (matrix/dimension-count m 1))
        dims (fn [m] [(row-size m) (col-size m)])
        dimensions
        (reduce
         (fn [ret m] (into ret (dims m)))
         [(:num-inputs net)]
         (matrices net))]
    (every? (fn [[x y]] (= x y)) (partition 2 (butlast dimensions)))))

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
        biases (mapv #(repeat % 0) dimensions)
        act-fns (mapv #(afs/get-fn (second %)) layers)
        deriv-fns (mapv #(afs/get-deriv (second %)) layers)]
    {:num-inputs num-inputs
     :layers (mapv (fn [m b a d] {:matrix m :bias b :act-fn a :deriv-fn d})
                   matrices biases act-fns deriv-fns)}))
