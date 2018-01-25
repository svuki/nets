(ns nets.examples
  (:require [nets.core :as core]
            [nets.net :as net]
            [nets.activation-functions :as act-fns]
            [nets.error-functions :as error-fns])
  (:gen-class))

(defn sample-input-fn
  []
  [(rand 1) (rand 1)])

(defn sample-output-fn
  [_]
  [1])

(defn always-one
  "In this exmaple, the neural net is trained to always output [1]."
  [learning-rate times-to-run]
  (let [test-net (net/new-net 2 [2 :sigmoid] [1 :sigmoid])]
    (core/multi-set-train-verbose
     test-net sample-input-fn sample-output-fn
     error-fns/l2 error-fns/l2-deriv learning-rate times-to-run)))
