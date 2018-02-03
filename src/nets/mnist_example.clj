
(ns nets.mnist-example
  (:require
            [nets.net :as net]
            [nets.backpropogation :as backprop]
            [nets.error-functions :as error-fns]
            [nets.printers :as printers]
            [clojure.core.matrix :as matrix])
  (:use nets.utils nets.matrix-utils))

;;; This file contains an example using the MNIST handwritten digit
;;; classification data set. The data is is located in external files,
;;; it may downloaded from http://yann.lecun.com/exdb/mnist/. There are four
;;; files: (1) a training set of 60,000 images, (2) labels for the training
;;; set, (3) a test set of 10,000 images, and (4) labels for the test set. The
;;; images are greyscale.

(def training-labels-stream
  (java.io.RandomAccessFile. "test/nets/train-labels.idx1-ubyte" "r"))
(def training-images-stream
  (java.io.RandomAccessFile. "test/nets/train-images.idx3-ubyte" "r"))
(def test-labels-stream
  (java.io.RandomAccessFile. "test/nets/t10k-labels.idx1-ubyte" "r"))
(def test-images-stream
  (java.io.RandomAccessFile. "test/nets/t10k-images.idx3-ubyte" "r"))


;;; The files are in the IDX file format. For our purposes the relevant data
;;; is as follows:
;;;     Label files:
;;;         Labels begin at offset 0008.
;;;         Each label is a single unisgned byte.
;;;         Label values range from 0 to 9.

(defn m
  "Temporary fix to account for wrong order of labels... Also this is why
  everything is incorrect from a black/white perspective..."
  [n]
  (if (= n 0)
    9
    (- (mod n (- 9)))))

(defn go-to-labels
  "Sets the filestream to the byte of the first label"
  [label-stream]
  (.seek label-stream 8))

(defn get-label
  "Returns the Nth label as an integer."
  [label-stream]
  (let [c (try (.read label-stream)
               (catch java.io.EOFException e
                 (do (go-to-labels label-stream)
                     (.read label-stream))))]
    c))

;;;     Image files:
;;;         Image data begins at offset 0016.
;;;         Each pixel is a single unsigned byte.
;;;         Pixel values range from 0 to 255.
;;;         Each image is a 28 by 28 pixels.
;;;         The data for a single image is 784 consecutive pixels values.

(defn go-to-images
  "Sets the filestream to the first pix data."
  [fstream]
  (.seek fstream 16))
(defn get-image
  "Returns the next image as a vector of 784 doubles. Each pixel value is divided
  by the maximum possible pixel value (255) to produce a number between 0 and 1."
  [fstream]
  (try  (matrix/array (repeatedly (* 28 28)
                                  (fn [] (/ (.read fstream)
                                            255.0))))
        (catch java.io.EOFException e
          (do (go-to-images fstream)
              (get-image fstream)))))


(def mnist-training-input-fn
  #(get-image training-images-stream))
;;; For now we use these function with core/train-for. That function specified
;;; that outputs take one argument, so we have to discard the argument here.
(def mnist-training-output-fn
  (fn [& rest] (one-hot (get-label training-labels-stream) 10)))
(def mnist-test-input-fn
  #(get-image test-images-stream))
(def mnist-test-output-fn
  (fn [& rest] (one-hot (get-label test-labels-stream) 10)))

(def mnist-tprofile
  {:lrate 0.1
   :cost-fn :cross-entropy
   :input-fn mnist-training-input-fn
   :output-fn mnist-training-output-fn})

(defn prediction-vector
  [v]
  (let [mx (apply max (seq v))]
    [(.indexOf (seq v) mx) mx]))

(defn wrong?
  [prediction-vec target]
  (let [target-index (.indexOf (seq target) 1.0)]
    (= target-index (first prediction-vec))))


(defn print-summary
  [output target]
  (let [pvec (prediction-vector output)]
    (printf "Predicts %d with confidence %.3f.\n" (first pvec) (second pvec))
    (printf "Actual: %d.\n" (.indexOf (seq  target) 1.0))))

(defn printer
  [output target]
  (print-summary output target)
  (newline)
  (printers/print-vec-comp output target))


(defn continue-prompt
  [tprofile]
  (if (y-or-n? "Continue training? (y/n)")
    (do (newline)
        (printf "How many interations? (integer) ")
        (flush)
        (let [iterations (Integer. (read-line))]
          (newline)
          (if (y-or-n? (format "The current learning rate is %f. Would you like to change it? (y/n) "
                               (float (:lrate tprofile))))
            (do (newline)
                (printf "Enter a new value: ")
                (flush)
                (let [new-lrate (Float. (read-line))]
                  [iterations new-lrate]))
            [iterations (:lrate tprofile)])))
    nil))

(use 'mikera.image.core)
(use 'mikera.image.colours)

(defn invert [val]
  (if (= 0 val)
    255
    (- (mod val (- 255)))))

(defn grey-to-rgb [g-val]
  (let [grey-val (invert g-val)] (rgb grey-val grey-val grey-val)))

(def digit-img (new-image 28 28))
(def pixels (get-pixels digit-img))

(defn show-img [img-data]
  (dotimes [i (* 28 28)]
    (aset pixels i (nth (map #(-> % (* 255.0) grey-to-rgb) img-data) i)))
  (set-pixels digit-img pixels)
  (show digit-img :zoom 5.0))

(defn show-digit [digit-data]
  (let [img (new-image 28 28)
        ps (get-pixels img)]
    (dotimes [i (* 28 28)]
      (aset ps i (nth (map grey-to-rgb digit-data) i)))
    (set-pixels img ps)
    (show img :zoom 10.0)))

(defn mnist-sample
  [net training-profile trials]
  (let [inputs  (repeatedly trials (:input-fn training-profile))
        output  (map #(backprop/net-eval net %) inputs)
        targets (map (:output-fn training-profile) (range trials))]
    (dorun (map printer output targets))))


(defn mnist-runner
  "A training runner for mnist dataset."
  [net training-profile iterations]
  (let [input ((:input-fn training-profile))
        output ((:output-fn training-profile) input)]
    ;(show-img (map #(* 255.0 %) input))
    (let [next-net (backprop/sgd net input output
                                 (:lrate training-profile)
                                 (error-fns/get-cost-grad
                                  (:cost-fn training-profile)))]
      (when (= 0 (mod iterations 500))
        (printf "Iterations remaining: %d.\n" iterations)
        (flush)
        (net/to-file next-net "mnist_net.txt"))
      (if (= 0 iterations)
        (do (mnist-sample net training-profile 10)
            (let [cont (continue-prompt training-profile)]
              (if cont
                (mnist-runner next-net
                              (assoc training-profile :lrate (second cont))
                              (first cont))
                (do (newline)
                    (printf "OK, exiting.\n")))))
        (recur next-net training-profile (dec iterations))))))

(defn mnist-example
  [iterations]
  (let [test-net (net/new-net (* 28 28) [400 :leaky-relu] [10 :softmax])]
    (.seek training-images-stream 0)
    (.seek training-labels-stream 0)
    (go-to-images training-images-stream)
    (go-to-labels training-labels-stream)
    (mnist-runner test-net mnist-tprofile iterations)))

(defn mnist-test
  [net]
  (go-to-images test-images-stream)
  (go-to-labels test-labels-stream)
  (let [preds (map (fn [_] (predict
                            (backprop/net-eval
                             net
                             (get-image test-images-stream))))
                   (range 10000))
        actuals (map (fn [_] (get-label test-labels-stream))
                     (range 10000))]
    (/ (reduce (fn [acc [p a]]
                 (if (= p (m a)) (inc acc) (do (printf "%d != %d\n" p (m a))
                                               (flush)
                                               acc)))
               0
               (map vector preds actuals))
       10000.0)))



(defn img-cycler
  [time-to-cycle]
  (let [digit-img (new-image 28 28)
        pixels (get-pixels digit-img)]
    (.seek training-images-stream 0)
    (go-to-images training-images-stream)
    (loop [img-data (get-image training-images-stream)]
      (dotimes [i (* 28 28)]
        (aset pixels i (nth (map grey-to-rgb img-data) i)))
      (set-pixels digit-img pixels)
      (show digit-img :zoom 10.0)
      (Thread/sleep time-to-cycle)
      (recur (get-image training-images-stream)))))


(defn rate->msecs
  [rate]
  (* 1000
     (/ 1.0 rate)))
(defn mnist-img-sample
  "Shows the specified image and the prediction, shows RATE images per second."
  [net rate]
  (go-to-images training-images-stream)
  (go-to-labels training-labels-stream)
  (loop
      [img-data  (mnist-training-input-fn)
       label-data    (mnist-training-output-fn)]
    (let [prediction (predict (backprop/net-eval net img-data))
          actual     (predict label-data)]
      (show-img img-data)
      (printf "Predicts: %d, actual: %d.\n" (m prediction) (m actual))
      (flush)
      (Thread/sleep (rate->msecs rate))
      (recur (mnist-test-input-fn) (mnist-test-output-fn)))))


(defn img-label
  []
  (.seek test-labels-stream 0)
  (go-to-labels test-labels-stream)
  (go-to-images test-images-stream)
  (loop []
    (show-img (mnist-test-input-fn))
    (printf "Is: %d.\n" (m (predict (mnist-test-output-fn))))
    (flush)
    (Thread/sleep 1000)
    (recur)))
