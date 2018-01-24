;;; This namespace provides a function NEW-NET which creates MLP based on the number of inputs, the number of layers, the neurons per layer, and the activation function for a given layer.
(ns nets.net
  (:require [nets.activation-functions :as afs]
            [clojure.algo.generic.math-functions :as math])
  (:gen-class))

(defn fill-matrix
  "Returns a matrix of dimension M rows and N columns populated with values obtained by calling FUNC. FUNC is a function of 0-arity."
  [num-rows num-cols func]
  (let [weights (take (* num-rows  num-cols) (repeatedly func))]
    (partition num-cols weights)))

; Note: this is ideal if the input is normalized
(defn weight-initializer
  "Returns a function which returns a random value between (-1/sqrt(d), 1/sqrt(d))"
  [d]
  (let [lim (/ 1 (math/sqrt d))]
    (fn [] (+ (- lim) (rand (* 2 lim))))))

;;; TODO, allow the user to pass in a custom matrix population function
;;; TODO, change matrix population functiont o be dependent on input...
(defn new-net
  "Returns a new net with NUM-INPUTS inputs. The remaining input are vectors of size two, the 0th value describing the amount of neurons in the layer and the 1th value a keyword corresponding to one of the activation function as specified in nets.activation-functions. Currently each layer is assumed to be fully connected and the weights are chosen to be between (-1/sqrt (d), 1/sqrt (d)) where d is the number of neurons in the preceding layer. In addition a bias vector will be initialized in all layers as the 0 vector.
  For example:
  (new-net 64 [10 :relu] [5 :sigmoid])"
  [num-inputs & layers]
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
