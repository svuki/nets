(ns nets.sgd-handler
  (require [nets.net :as net])
  (use nets.utils.utils))

(defn sgd
  "Runs the SGD computation in the current thread."
  [net-desc tprofile iterations impl-IF]
  ((:run-sgd impl-IF) net-desc tprofile iterations))

(defn- sgd-handler-helper
  [net-atom iters-atom tprofile imp-IF]
  (fn [iters]
    (swap! iters-atom #(- % iters))
    (swap! net-atom
           (fn [net]
             (sgd net tprofile iters imp-IF)))))

(defn- iter-chunks [n i]
  (if (<= n i)
    (list n)
    (lazy-seq (cons i (iter-chunks (- n i) i)))))

; TODO: improve time estimation. Look into making sgd-handler repl like
(defn sgd-handler
  "Runs the specified SGD computation in a seperate thread."
  [net-desc tprofile iterations impl-IF]
  (let [net      (atom net-desc)
        time-est (promise)
        iters    (atom iterations)
        is       (iter-chunks iterations 50)
        proc     (sgd-handler-helper
                  net iters tprofile impl-IF)
        computation
        (future
          (do
            (deliver time-est (time-sec (proc (first is))))
            (dorun (map proc (rest is)))
            (println "The computation has finished.")))]
    (fn [kw & args]
      (case kw
        :deref      @computation
        :net        @net
        :done?      (future-done? computation)
        :save       (apply (partial net/to-file @net) args)
        :iterations @iters
        :time       (if (and (realized? time-est) @time-est)
                      (printf "Estimated time remaining: %.2f seconds.\n"
                              (* @time-est (/ @iters 50.0)))
                      (println "Still Estiming...\n"))
        (println "Invalid option. Options are :net, :done?, :save, :iterations, :time")))))
        
                                
