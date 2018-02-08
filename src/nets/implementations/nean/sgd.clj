(ns nets.implementations.nean.sgd
  (:use uncomplicate.neanderthal.core
        uncomplicate.neanderthal.native
        uncomplicate.fluokitten.core
        uncomplicate.neanderthal.vect-math))

; When is it appropraite to use the destructive vaiants mv! and axpy!?
; Whenever we don't need to reuse v after...

(def ^:dynamic *smax-cent?* false) 

(defn net-eval
  ([[ms bs afs _] v]
   (net-eval ms bs afs v)) 
  ([ms bs afs v]
   (if (empty? ms)
     v
     (recur (rest ms)
            (rest bs)
            (rest afs)
            ((first afs)
             (axpy (first bs)
                   (mv (first ms) v)))))))

(defn fprop
  "Propgate input V through the net. Returns a vector
  [P-VALS A-VALS L-OUT] where P-vals are the preactivation values
  of each layer, a-vals are the activation values of non-output layers
  and l-out is the net output (activation value of the output lyaer)."
  [[ms bs] afs v]
  {:pre [(= (dim v) (ncols (first ms)))]
   :post [(= (count ms)
             (count (first %))
             (count (second %)))]}
  ; p-vals = preactivation values
  ; a-vals = activation values (layer-outputs)
  ; l-out  = last layer output (last activation value)
  (loop [p-vals []
         a-vals []
         ms     ms
         bs     bs
         afs    afs
         l-out  v]
    (if (empty? ms)
      [p-vals a-vals l-out]
      (let [p-val (axpy (mv (first ms) l-out)
                        (first bs))]
        (recur
         (conj p-vals p-val)
         (conj a-vals l-out)
         (rest ms)
         (rest bs)
         (rest afs)
         ((first afs) p-val))))))

(defn- weight-gradients
  [deltas a-vals]
  [deltas (mapv rk deltas a-vals)])

(defn bprop
  [[p-vals a-vals output] mats target dfs cgrad]
  (let [init-error (if *smax-cent?*
                     (axpy -1.0 target output)
                     (mul ((last dfs) (last p-vals))
                           (cgrad output target)))]
    (loop [ms mats
           dp-vals (mapv #(%1 %2) (butlast dfs) (butlast p-vals))
           deltas [init-error]]
      (if (empty? dp-vals)
        (weight-gradients (reverse deltas) a-vals)
        (recur
         (butlast ms)
         (butlast dp-vals)
         (conj deltas
               (mul (last dp-vals)
                    (mv (trans (last ms))
                        (last deltas)))))))))


(defn weight-update
  [[deltas matrix-grads] ms biases lrate]
    [(mapv #(axpy (- lrate) %1 %2)
           matrix-grads ms)
     (mapv #(axpy (- lrate) %1 %2)
           deltas biases)])

(defn sgd
  [[ms bs afs dfs] training-pairs lrate cgrad iterations]
  ; TODO: benchmark using transients vs not. In fn update we cannot 
  ; use fmap! so a copy of ms and bs is made?
  (loop [mbs    [ms bs]
         tdata  training-pairs
         iters iterations]
    (if (= 0 iters)
      [(first mbs) (second mbs)]
      (let [inp  (first (first tdata))
            targ (last  (first tdata))
            ms   (first mbs)
            bs   (second mbs)]
        (recur
         (-> mbs
             (fprop afs inp)
             (bprop ms targ dfs cgrad)
             (weight-update ms bs lrate))
         (rest tdata)
         (dec iters))))))
  
