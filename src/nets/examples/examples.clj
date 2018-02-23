(ns nets.examples
  (:require [nets.trainers :as trainers]
            [nets.net :as net]
            [nets.activation-functions :as act-fns]
            [nets.error-functions :as error-fns]
            [clojure.algo.generic.math-functions :as math]
            [nets.backpropogation :as backprop])
  (:gen-class))


;;; Example 1: Given an input of size 2, always return 1.

(defn sample-input-fn
  []
  [(rand 1) (rand 1)])

(defn sample-output-fn
  [_]
  [1])

(def always-one-tprofile
  {:lrate 0.2 :cost-fn :mean-squared
   :input-fn sample-input-fn :output-fn sample-output-fn})

(defn always-one
  "In this exmaple, the net is trained to always output [1]."
  [times-to-run]
  (let [test-net (net/new-net 2 [10 :relu] [1 :sigmoid])]
    (trainers/train-for test-net always-one-tprofile times-to-run)))

(defn reverse-fn
  [[x y]] [y x])

(def always-rev-tprofile
  {:lrate 0.05 :cost-fn :mean-squared
   :input-fn sample-input-fn :output-fn reverse-fn})

(defn always-reverse
  "In this example the net is trained to learn the function
  [x y] --> [y x]."
  [iterations]
  (let [test-net (net/new-net 2 [20 :relu] [2 :sigmoid])]
    (trainers/train-for test-net always-rev-tprofile iterations)))

(defn sin-input-fn
  []
  [(rand (* 2 Math/PI))])

(defn sin-output-fn
  "Normalized sin function to [0,1]. "
  [[x]]
  [(* 0.5 (inc (Math/sin x)))])

(def sin-tprofile
  {:lrate 0.2 :cost-fn :mean-squared
   :input-fn sin-input-fn :output-fn sin-output-fn})

(defn sin-example
  [iterations]
  (let [test-net (net/new-net 1 [40 :relu] [1 :sigmoid])]
    (trainers/train-for test-net sin-tprofile iterations)))


;;; Floor function
(defn floor-input
  [] [(/ (rand 5) 5.0)])
(defn floor-output
  [[y]]
  (let [index (int (Math/floor (* y 5.0)))]
    ; We want to use the softmax activation function and
    ; cross entropy as the cost function, so we'll treat
    ; the target output as a vector of size 5 with a 1
    ; at position index and 0s everywhere else
    (if (= 0 index)
      [1.0 0.0 0.0 0.0 0.0]
      (into (conj (vec (take index (repeat 0.0))) 1.0)
            (vec (take (dec (- 5 index)) (repeat 0.0)))))))

(def floor-tprofile
  {:lrate 0.2 :cost-fn :cross-entropy
   :input-fn floor-input :output-fn floor-output})

(defn floor-example
  [iterations]
  (let [test-net (net/new-net 1 [30 :relu] [5 :softmax])]
    (trainers/train-for test-net floor-tprofile iterations)))
