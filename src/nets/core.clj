(ns nets.core
  (:require [nets.examples :as ex]))

(defn -main
  []
  (printf "In the future this will provide an interactive environment for constructing and training nets.
For Now all it does is compute the sin function. I suggest running it for 10,000+ iterations. Enjoy\n")
  (flush)
  (ex/sin-example 1))
