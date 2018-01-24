(defproject nets "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.clojure/algo.generic "0.1.2"]
                 [net.mikera/core.matrix "0.61.0"]
                 [org.clojure/math.numeric-tower "0.0.4"]]
  :main ^:skip-aot nets.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
