(ns nets.core
  (:require [nets.examples :as ex]
            [nets.mnist-example :as mnist]
            [nets.benchmark :as bench]))

(defn -main
  ([]
   (bench/run-bench))
  ([arg]
   (mnist/mnist-example (Integer. arg))))
