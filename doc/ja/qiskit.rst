Qiskit 概要
===========

設計思想
--------

QISKitは、深さが限定的な量子回路を設計して直近の応用を意識した量子コンピューターのアプリケーションを開発したり
実験を行うためのソフトウェア開発キットです。QISKitの中では量子プログラムは量子回路の配列として管理されています。
プログラムのフローは「ビルド」、「コンパイル」、「実行」の3つの段階で構成されています。
「ビルド」ではユーザーが解きたい問題に対応する量子回路を作ります。
「コンパイル」は量子回路をバックエンドで実行するための変換を行います。
バックエンドは量子回路を実行する実体で、シミュレーターや実機の選択肢があります。
実機のバックエンドも、量子ビット数やフィディレティーやQuantum Volume等が異なるものがあり、
コンパイルはそのような制限のあるバックエンドに合わせて量子回路の変換を行います。
「実行」はジョブを生成します。ジョブが終了すると結果のデータを得ることができます。
プログラムによっては、このデータをまとめる方法もあります。 これはあなたが望む答えを提供するか、
次のインスタンスのためのより良いプログラムを作ることを可能にします。


プロジェクト概要
----------------
QISKitプロジェクトの構成は以下の通りです:

* `Qiskit Terra <https://github.com/Qiskit/qiskit-terra>`_: 
　量子コンピューティングの実験、プログラム、アプリケーションを書くためのPython のサイエンス開発キット

* `Qiskit Aqua <https://github.com/Qiskit/aqua>`_:  
　ノイズのある中規模量子コンピューター(NISQ) のためのアプリケーションを構築するライブラリーとツール

* `Qiskit SDK <https://github.com/Qiskit/qiskit-terra>`_:
  量子コンピューターのプログラムやアプリケーションを開発して実験を行うためのPythonソフトウェア開発キット

* `Qiskit API <https://github.com/IBM/qiskit-api-py>`_:
  IBM Q ExperienceのHTTP APIのPythonラッパー。
  IBM Q Experienceを通じて量子プログラムの実行が可能。

* `Qiskit OpenQASM <https://github.com/IBM/qiskit-openqasm>`_:
  OpenQASMという中間言語の仕様書、例、マニュアルおよびツール群。

* `Qiskit Tutorial <https://github.com/IBM/qiskit-tutorial>`_:
  Jupyterノートブックで書かれたQISKitのチュートリアル集。
