(ns nets.core
  (:require [nets.examples :as ex]))

(defn -main
  [ iterations ]
  (printf "In the future this will provide an interactive environment for constructing and training nets.
           For Now all it does is compute the floor function. Enjoy\n")
  (ex/floor-example 1000))
