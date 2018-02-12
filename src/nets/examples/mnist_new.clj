(ns nets.mnist-new
  (:require
   [nets.net :as net]
   [nets.implementations.nean.interface :as inter]
   [nets.sgd-handler :as handler])
  (:use
   nets.utils
   nets.matrix-utils))

(def training-labels-stream
  (java.io.RandomAccessFile. "test/nets/train-labels.idx1-ubyte" "r"))
(def training-images-stream
  (java.io.RandomAccessFile. "test/nets/train-images.idx3-ubyte" "r"))
(def test-labels-stream
  (java.io.RandomAccessFile. "test/nets/t10k-labels.idx1-ubyte" "r"))
(def test-images-stream
  (java.io.RandomAccessFile. "test/nets/t10k-images.idx3-ubyte" "r"))

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



(defn go-to-images
  "Sets the filestream to the first pix data."
  [fstream]
  (.seek fstream 16))

(defn get-image
  "Returns the next image as a vector of 784 doubles. Each pixel value is divided
  by the maximum possible pixel value (255) to produce a number between 0 and 1."
  [fstream]
  (doall (repeatedly (* 28 28) (fn [] (/ (.read fstream) 255.0)))))

(def train-inp-fn
  #(get-image training-images-stream))
;;; For now we use these function with core/train-for. That function specified
;;; that outputs take one argument, so we have to discard the argument here.
(def train-out-fn
  (fn [& rest] (one-hot (get-label training-labels-stream) 10)))
(def mnist-test-input-fn
  #(get-image test-images-stream))
(def mnist-test-output-fn
  (fn [& rest] (one-hot (get-label test-labels-stream) 10)))

(defn reset-data
  []
  (go-to-images training-images-stream)
  (go-to-labels training-labels-stream))
(reset-data)
(def tdata (repeatedly 60000 (fn [] (vector (train-inp-fn) (train-out-fn)))))

(def mnist-tprofile
  {:lrate 0.1
   :cost-fn :cross-entropy
   :training-data tdata})

(defn mnist-example
   [iterations]
  (let [test-net (net/new-net (* 28 28) [400 :leaky-relu] [10 :softmax])]
    (go-to-images training-images-stream)
    (go-to-labels training-labels-stream)
    (handler/sgd test-net
                 mnist-tprofile
                 iterations
                 inter/interface)))


