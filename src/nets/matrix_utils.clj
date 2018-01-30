(ns nets.matrix-utils)

(defn one-hot
  "Returns a vector of size M with all zeros except for a 1 at
  index INDEX."
  [index m]
  (into (conj (take index (repeat 0.0)) 1.0)
        (take (- (dec m) index) (repeat 0.0))))

(defn kdelta
  "The kronecker delta. Returns 1.0 if i == j and 0.0 otherwise."
  [i j]
  (if (= i j) 1.0 0.0))

(defn make-matrix
  "Takes integer M, integer N, and a function F of two arguments.
  Produces a matrix of M rows and N columns such that the (i,j) entry
  is (F i j)"
  [m n f]
  (let [mat-indices (map (fn [i] (map (fn [j] [i j]) (range n))) (range m))]
    (mapv #(mapv (fn [[i j]] (f i j)) %) mat-indices)))
