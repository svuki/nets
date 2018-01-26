(ns nets.examples
  (:require [nets.core :as core]
            [nets.net :as net]
            [nets.activation-functions :as act-fns]
            [nets.error-functions :as error-fns]
            [clojure.algo.generic.math-functions :as math]
            [nets.backpropogation :as backprop])
  (:gen-class))

(defn sample-input-fn
  []
  [(rand 1) (rand 1)])

(defn sample-output-fn
  [_]
  [1])

(def always-one-tprofile
  {:lrate 0.2 :err-fn error-fns/l2 :err-deriv error-fns/l2-deriv
   :input-fn sample-input-fn :output-fn sample-output-fn})
(defn always-one
  "In this exmaple, the net is trained to always output [1]."
  [times-to-run]
  (let [test-net (net/new-net 2 [10 :sigmoid] [1 :sigmoid])]
    (core/train-for test-net always-one-tprofile times-to-run)))

(defn reverse-fn
  [[x y]] [y x])

(def always-rev-tprofile
  {:lrate 0.05 :err-fn error-fns/l2 :err-deriv error-fns/l2-deriv
   :input-fn sample-input-fn :output-fn reverse-fn})

(defn always-reverse
  "In this example the net is trained to learn the function
  [x y] --> [y x]."
  [iterations]
  (let [test-net (net/new-net 2 [10 :sigmoid] [2 :sigmoid])]
    (core/train-for test-net always-rev-tprofile iterations)))

(defn sin-input-fn
  []
  [(rand (* 2 Math/PI))])
(defn sin-output-fn
  "Normalized sin function to [0,1]. "
  [[x]]
  [(* 0.5 (inc (Math/sin x)))])

(defn good-enough?
  "Returns true if the average error over 100 trials is less than 5%"
  [net input-fn target-fn error-fn]
  (let [inputs (take 100 (repeatedly input-fn))
        targets (mapv #(target-fn %) inputs)
        outputs (mapv #(backprop/net-eval net %) inputs)
        errors (mapv #(error-fn %1 %2) outputs targets)
        ; hacky: what we want is the norm of the targets, but this works
        perrors (mapv #(/ %1 (error-fn %2 0.0)) errors targets)]
    (/ (reduce + perrors) 100)))

(defn sin-example
  []
  (let [net (net/new-net 1 [20 :sigmoid] [1 :sigmoid])]
    (core/multi-set-train-verbose net sin-input-fn sin-output-fn
                                  error-fns/l2 error-fns/l2-deriv 0.1 100)))
