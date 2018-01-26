(ns nets.utils
  (:gen-class))

(defmacro showlet
  "Performs a let binding as usual, but prints out the value of each binding immediately
  after the binding succeeds."
  [letvec body]
  (if (empty? letvec)
    body
    (let [[symbol expr & r] letvec]
      `(let [~symbol ~expr]
         (do (print ~(name symbol) ~symbol)
             (newline)
             (showlet ~r ~body))))))
