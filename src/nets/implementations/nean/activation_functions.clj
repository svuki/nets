(ns nets.implementations.nean.activation-functions
  (:require [uncomplicate.neanderthal.vect-math :as vmath])
  (:use uncomplicate.fluokitten.core
        uncomplicate.neanderthal.native
        uncomplicate.neanderthal.core))

; Activation function sused by the neanderthal implementation

(defn- vectorize
  "Takes a scalar function and returns its vectorized form."
  [func]
  (fn [v] (fmap func v)))

(defn- sigmoid ^double
  [^double x]
  (/ 1.0 (+ 1 (Math/exp (- x)))))

(defn- sigmoid-deriv ^double
  [^double x]
  (* (sigmoid x) (- 1 (sigmoid x))))

(defn- tanh
  [v]
  (vmath/tanh v))

(defn- tanh-deriv ^double
  [^double x]
  (- 1 (Math/pow (sigmoid x) 2)))

(defn- relu
  [v]
  (vmath/fmax v (zero v)))

(defn- relu-deriv ^double
  [^double x]
  (if (> x 0) 1.0 0.0))

(defn- leaky-relu ^double
  [^double x]
  (max 0 x))

(defn- leaky-relu-deriv ^double
  [^double x]
  (cond (> x 0) 1.0
        (<= x 0) (* 0.0001 x)))

(defn- softplus ^double
  [^double x]
  (Math/log (+ 1.0 (Math/exp x))))

(defn- softplus-grad ^double
  [^double x]
  (/ 1.0
     (+ 1 (Math/log (- x)))))

(defn- vmax ^double
  [v]
  (entry v (imax v)))

(defn- m- ^double
  [^double x ^double y]
  (- x y))

(defn- softmax
  [v]
  ; to avoid overflows we subtract the largest x from each of the values
  ; before exponentiating. This does not change the result.
  (let [exps (vmath/exp (fmap m- v (dv (repeat (dim v) (vmax v))))) ; TODO : this is ugly
        s    (sum exps)]
    (scal (/ 1.0 s) exps)))

;; TODO: sloppy. In what cases would we use softmax without cross entropy? Would we ever use
;; it in a hidden layer? Right now cross-entropy/softmax on output is handled by a seperate
;; routine so the softmax-jacobian is never calculated
(defn- softmax-jacobian
  "Given a vector v, returns the tranpose of the jacobian matrix
  of the softmax function evaluated at v."
  [v]
  ; To keep with the convention that we multiply by row vectors on the left
  ; we transpose the jacboian so that the first column gives the partial
  ; derivatives of the first component function of softmax with respect
  ; to the the inputs v_0 .. v_n.
  (throw (Exception. "The softmax-jacobian has not yet been implemented.
You can still use the softmax function, but only combined with the cross
entropy error function.\n")))


(def act-fns
  {:sigmoid (mapv vectorize [sigmoid sigmoid-deriv])
   :tanh [tanh (vectorize tanh-deriv)]
   :relu [relu (vectorize relu-deriv)]
   :leaky-relu (mapv vectorize [leaky-relu leaky-relu-deriv])
   :softmax [softmax softmax-jacobian]
   :softplus (mapv vectorize [softplus softplus-grad])})

(defn get-fn
  "Returns the function associated with NAME. Returns nil if no function is found."
  [name]
  (first (get act-fns name)))

; TODO: function should be renamed to get-gradient or get-grad
(defn get-deriv
  "Returns the derivativ eof the function associated with NAME.
  Returns nil if no function is found."
  [name]
  (second (get act-fns name)))
