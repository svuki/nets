(ns nets.core
  (:require [nets.mnist-new :as mnist]
            [criterium.core :as crit]))
  
(defn -main
  []
  (time (mnist/mnist-example 10000)))
