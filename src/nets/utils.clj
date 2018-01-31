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

(defn prompt-read
  [prompt]
  (printf "%s: " prompt)
  (flush)
  (read-line))

(defn y-or-n?
  [prompt]
  (= "y"
     (loop []
       (or
        (re-matches #"[yn]" (.toLowerCase (prompt-read prompt)))
        (do (newline)
            (recur))))))

(defn no-NaN?
  "Ensures x is not NaN"
  [x]
  (not (Double/isNaN x)))
(defn no-NaNs?
  "Ensures no element in v is NaN"
  [v]
  (not-any? #(Double/isNaN %) v))
