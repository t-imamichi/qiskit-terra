ライブラリの構成
================

プログラミング インターフェース
-------------------------------

*qiskit* ディレクトリがメインのPythonモジュールで
:py:class:`QuantumProgram <qiskit.QuantumProgram>`,
:py:class:`ClassicalRegister <qiskit.ClassicalRegister>`,
:py:class:`QuantumCircuit <qiskit.QuantumCircuit>` のインターフェースを含みます。

実行手順は次の通りです。ユーザーは *QuantumProgram* （量子プログラム）を作成しそれに
複数の量子回路の生成、変更、コンパイル、と実行ができます。
各 *QuantumCircuit* （量子回路）には *QuantumRegister* （量子レジスター）か
*ClassicalRegister* （古典レジスター）があります。
これらのオブジェクトのメソッドを通じて、量子ゲートを適用して量子回路を定義します。
*QuantumCircuit* は **OpenQASM** コードを出力することができます。
このコードは *qiskit* ディレクトリの他のコンポーネントに流すことができます。

:py:mod:`extensions <qiskit.extensions>` ディレクトリは他の量子ゲートやアルゴリズムのサポートに必要な
量子回路の拡張のモジュールを含みます。
現在は典型的な量子ゲートを定義した :py:mod:`standard <qiskit.extensions.standard>` 拡張と
２つの追加の拡張モジュールがあります：
:py:mod:`qasm_simulator_cpp <qiskit.extensions.simulator>` と
:py:mod:`quantum_initializer <qiskit.extensions.quantum_initializer>`.


内部モジュール
--------------

以下のディレクトリは開発中の内部モジュールを含みます:

- :py:mod:`qasm <qiskit.qasm>` モジュールは **OpenQASM** ファイルを解析します。
- :py:mod:`unroll <qiskit.unroll>` モジュールはターゲットのゲートに応じて **OpenQASM** の翻訳と展開(unroll)を行います。
  (ゲート文のサブルーチンやループの展開も必要に応じて行います)
- :py:mod:`dagcircuit <qiskit.dagcircuit>` モジュールは量子回路をグラフとして処理します。
- :py:mod:`mapper <qiskit.mapper>` モジュールは、量子回路をカップリング（直接操作可能な量子ビットのペア）の制限がないバックエンドから制限のある実機で実行するために量子回路の変換を行います。
- :py:mod:`backends <qiskit.backends>` モジュールは量子回路のシミュレーターを含みます。
- *tools* ディレクトリはアプリケーション、分析、可視化のメソッドを含みます。

量子回路は以下の様にコンポーネントの間を流れます。
プログラミングインターフェースを用いて生成した **OpenQASM** 量子回路はテキストか *QuantumCircuit* オブジェクトです。
**OpenQASM** ソースコードはファイルか文字列で、 *Qasm* オブジェクトに渡されます。
そして、ソースコードはparseメソッドで分析されて抽象構文木(abstract syntax tree, **AST**)が生成されます。
**AST** は *Unroller* に渡されます。*Unroller* は異なる *UnrollerBackend* のいずれかを取り付けることができます。
テキストを出力する *PrinterBackend*、シミュレーターと実機のバックエンドの入力を生成する *JsonBackend*、
*DAGCircuit* を生成する *DAGBackend*、 *QuantumCircuit* オブジェクトを生成する *CircuitBackend* があります。
*DAGCircuit* は「展開された」 **OpenQASM** の回路を有向非巡回グラフ(directed acyclic graph, DAG)として持ちます。
*DAGCircuit* は回路構成の表現、変換、性質の計算、結果を **OpenQASM** で出力するメソッドがあります。
この全体の流れは *mapper* モジュールが量子回路を、 *CouplingGraph* で定義されたカップリング（直接操作可能な量子ビットのペア）の実機で実行するために変換するために行われます。
このコンポーネントの構成は将来的に変更の可能性があります。

量子回路が現在のモジュールでどのように変換されるかを以下の図にまとめます。



.. image:: ../../images/circuit_representations.png
    :width: 600px
    :align: center

いくつかのunrollerバックエンドとその出力を以下にまとめます:



.. image:: ../../images/unroller_backends.png
    :width: 600px
    :align: center


ログの記録
---------

Terra は、標準のPython "logging"ライブラリー
<https://docs.python.org/3/library/logging.html>`_ を使用して、"`qiskit.*`" ロガーのファミリーで、
複数のメッセージを出力し、ログレベルの標準的な規則に従います：

.. tabularcolumns:: |l|L|

+--------------+----------------------------------------------+
| レベル        | 使用時                                        |
+==============+==============================================+
| ``DEBUG``    | 詳細な情報。通常、問題の診断時にのみ重要です。       |
|              |                                              |
+--------------+----------------------------------------------+
| ``INFO``     | 物事が期待どおりに働いていることの確認。            |
|              |                                              |
+--------------+----------------------------------------------+
| ``WARNING``  | 予期せぬことが起こったか、または近い将来に何らかの   |
|              | 問題が発生したことの兆候    （「ディスクスペース不足」|   
|              | など）。 ソフトウェアは期待どおりに動作しています。   |
|              |                                              |
+--------------+----------------------------------------------+
| ``ERROR``    | より深刻な問題のため、ソフトウェアは何らかの機能を   |
|              | 実行できませんでした。                          |
+--------------+----------------------------------------------+
| ``CRITICAL`` | 重大なエラー。プログラム自体が実行を継続できない     |
|              | 可能性があることを示します。                      |
+--------------+----------------------------------------------+


便宜上、ハンドラーとqiskitロガーのレベルを変更する2つのメソッドが
:py:mod<`qiskit_logging.py`>: (:py:func:<`set_qiskit_logger()>` と
:py:func:<`unset_qiskit_logger`>) に用意されています。 これらの方法を使用すると、
環境のグローバルログ設定が妨げられる可能性があります。
Terraの上にアプリケーションを開発する場合は、考慮してください。

ログメッセージを出力するため、**logger** という名前のモジュールにグローバル変数が宣言されています。
この変数には、そのモジュールの **__name__** を持つロガーが含まれていて、メッセージの出力に使われます。 
たとえば、モジュールが `qiskit/some/module.py` の場合：

.. code-block:: python

   import logging

   logger = logging.getLogger(__name__)  # logger for "qiskit.some.module"
   ...
   logger.info("This is an info message)


テストする
--------

Terra は、uses the `standard Pyton "unittest" framework
<https://docs.python.org/3/library/unittest.html>`_ for the testing of the
different components and functionality.

As our build system is based on CMake, we need to perform what is called an
"out-of-source" build before running the tests.
This is as simple as executing these commands:

Linux and Mac:

.. code-block:: bash

    $ mkdir out
    $ cd out
    out$ cmake ..
    out$ make

Windows:

.. code-block:: bash

    C:\..\> mkdir out
    C:\..\> cd out
    C:\..\out> cmake -DUSER_LIB_PATH=C:\path\to\mingw64\lib\libpthreads.a -G "MinGW Makefiles" ..
    C:\..\out> make

This will generate all needed binaries for your specific platform.

For executing the tests, a ``make test`` target is available.
The execution of the tests (both via the make target and during manual invocation)
takes into account the ``LOG_LEVEL`` environment variable. If present, a ``.log``
file will be created on the test directory with the output of the log calls, which
will also be printed to stdout. You can adjust the verbosity via the content
of that variable, for example:

Linux and Mac:

.. code-block:: bash

    $ cd out
    out$ LOG_LEVEL="DEBUG" ARGS="-V" make test

Windows:

.. code-block:: bash

    $ cd out
    C:\..\out> set LOG_LEVEL="DEBUG"
    C:\..\out> set ARGS="-V"
    C:\..\out> make test

For executing a simple python test manually, we don't need to change the directory
to ``out``, just run this command:


Linux and Mac:

.. code-block:: bash

    $ LOG_LEVEL=INFO python -m unittest test/python/test_apps.py

Windows:

.. code-block:: bash

    C:\..\> set LOG_LEVEL="INFO"
    C:\..\> python -m unittest test/python/test_apps.py

Testing options
^^^^^^^^^^^^^^^

By default, and if there is no user credentials available, the tests that require online access are run with recorded (mocked) information. This is, the remote requests are replayed from a ``test/cassettes`` and not real HTTP requests is generated.
If user credentials are found, in that cases it use them to make the network requests.

How and which tests are executed is controlled by a environment variable ``QISKIT_TESTS``. The options are (where ``uc_available = True`` if the user credentials are available, and ``False`` otherwise): 

+-------------------+--------------------------------------------------------------------------------------------------------------------+-----------------------+--------------------------------------------------+
|  Option           | Description                                                                                                        | Default               |  If ``True``, forces                             |
+===================+====================================================================================================================+=======================+==================================================+
| ``skip_online``   | Skips tests that require remote requests (also, no mocked information is used). Does not require user credentials. | ``False``             | ``rec = False``                                  |
+-------------------+--------------------------------------------------------------------------------------------------------------------+-----------------------+--------------------------------------------------+
| ``mock_online``   | It runs the online tests using mocked information. Does not require user credentials.                              | ``not uc_available``  | ``skip_online = False``                          |
+-------------------+--------------------------------------------------------------------------------------------------------------------+-----------------------+--------------------------------------------------+
| ``run_slow``      | It runs tests tagged as *slow*.                                                                                    | ``False``             |                                                  |
+-------------------+--------------------------------------------------------------------------------------------------------------------+-----------------------+--------------------------------------------------+
| ``rec``           | It records the remote requests. It requires user credentials.                                                      | ``False``             | ``skip_online = False``                          |
|                   |                                                                                                                    |                       | ``run_slow = False``                             |
+-------------------+--------------------------------------------------------------------------------------------------------------------+-----------------------+--------------------------------------------------+

It is possible to provide more than one option separated with commas.
The order of precedence in the options is right to left. For example, ``QISKIT_TESTS=skip_online,rec`` will set the options as ``skip_online == False`` and ``rec == True``.	
