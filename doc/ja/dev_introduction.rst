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
| Level        | When it's used                               |
+==============+==============================================+
| ``DEBUG``    | Detailed information, typically of interest  |
|              | only when diagnosing problems.               |
+--------------+----------------------------------------------+
| ``INFO``     | Confirmation that things are working as      |
|              | expected.                                    |
+--------------+----------------------------------------------+
| ``WARNING``  | An indication that something unexpected      |
|              | happened, or indicative of some problem in   |
|              | the near future (e.g. 'disk space low').     |
|              | The software is still working as expected.   |
+--------------+----------------------------------------------+
| ``ERROR``    | Due to a more serious problem, the software  |
|              | has not been able to perform some function.  |
+--------------+----------------------------------------------+
| ``CRITICAL`` | A serious error, indicating that the program |
|              | itself may be unable to continue running.    |
+--------------+----------------------------------------------+

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

Terra は、異なるコンポーネントや機能をテストするために `標準の Pyton "unittest" フレームワーク
<https://docs.python.org/3/library/unittest.html>`_ を使います。

ビルドシステムはCMakeに基づいているので、テストを実行する前に「ソース外」ビルドを実行する必要があります。
これは、次のコマンドを実行するだけです:

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

これにより、特定のプラットフォームに必要なすべてのバイナリが生成されます。

テストを実行するために、 ``make test``のターゲットが利用可能です。
テストの実行（makeターゲット経由と手動起動ともに）は、環境変数 ``LOG_LEVEL``を考慮に入れます。
存在する場合、テストディレクトリに ``.log`` ファイルが作成され、ログ呼び出しの出力が生成されます。
これもstdoutに出力されます。その変数の内容を使って冗長性を調整することができます。例えば：

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

単純なPythonテストを手動で実行するには、ディレクトリーを ``out`` に変更する必要はありません。
このコマンドを実行するだけです：

Linux and Mac:

.. code-block:: bash

    $ LOG_LEVEL=INFO python -m unittest test/python/test_apps.py

Windows:

.. code-block:: bash

    C:\..\> set LOG_LEVEL="INFO"
    C:\..\> python -m unittest test/python/test_apps.py

テストのオプション
^^^^^^^^^^^^^^^

デフォルトでは、利用可能なユーザーの資格情報がない場合、オンラインアクセスが必要なテストは、記録された（模擬された）情報で実行されます。
つまり、リモートリクエストは ``test/cassettes`` から再生され、実際のHTTPリクエストは生成されません。
ユーザーの資格情報が見つかった場合は、ネットワークのリクエストが行われます。

どのようにしてどのテストが実行されるかは、環境変数 ``QISKIT_TESTS`` によって制御されます。 
オプションは次のとおりです（ユーザー資格情報が利用可能な場合は ``uc_available = True``、そうでない場合は ``False``）:

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

コンマで区切られた複数のオプションを指定することは可能です。
オプションの優先順位は、右から左です。 
たとえば、``QISKIT_TESTS=skip_online,rec`` は ``skip_online == False``と ``rec == True`` のオプションを設定します。
