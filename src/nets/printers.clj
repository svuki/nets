(ns nets.printers
  (:require [clojure.pprint]))



(defn to-string
  "Converts floats into strings of N decimal length. Converts
  vectors of numbers into a single string space seperated string."
  [n x]
  (let [fstring (str "%." (format "%d" n) "f")]
    (if (not (number? x))
      (apply str
             (interpose
              " "
              (mapv #(format fstring %) (mapv float x))))
      (format fstring (float x)))))

(defn sample-printer
  "Given vectors INPUTS, OUTPUTS, TARGETS, and ERRORS, prints a table with
  input | output | target | error on each row. This works poorly for
  large vectors."
  [inputs outputs targets errors]
  (let [float-formatter (partial to-string 5)
        strings (mapv #(mapv float-formatter %) [inputs outputs targets errors])]
    (clojure.pprint/print-table
     (apply
      (partial mapv #(zipmap ["INPUT" "OUTPUT" "TARGET" "ERROR"]
                             (vector %1 %2 %3 %4)))
      strings))))

(defn prinsep
  []
  (printf "-----------------------------------------------------------")
  (newline))

(defn print-vec-comp
  "Prints a side by side comparison of the elements of vector v1 and v2."
  [v1 v2]
  {:pre [(= (count v1) (count v2))]}
  (if (empty? v1)
    (prinsep)
    (do (printf "%.5f   " (first v1))
        (printf "%.5f   " (first v2))
        (newline)
        (recur (rest v1) (rest v2)))))
