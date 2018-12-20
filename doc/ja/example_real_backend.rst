実機バックエンドの例
^^^^^^^^^^^^^^^^^^^^
次のコードは、実際の量子デバイスで量子プログラムを実行する方法の例です。

.. code-block:: python

    # Qiskit Terraのインポート
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit import execute, IBMQ

    #  APIトークンの設定
    #  トークンは次のURLから入手可能です。https://quantumexperience.ng.bluemix.net/qx/account,
    #  "Personal Access Token" のセクションを参照してください。
    QX_TOKEN = "API_TOKEN"
    QX_URL = "https://quantumexperience.ng.bluemix.net/api"

    # オンラインのデバイスを使うためのIBM Q APIの認証
    # APIトークンとQXのURLが必要です。
    IBMQ.enable_account(QX_TOKEN, QX_URL)
    
    # 量子ビットの量子レジスターを生成
    q = QuantumRegister(2)
    # 2ビットの古典レジスターを生成
    c = ClassicalRegister(2)
    # 量子回路を生成
    qc = QuantumCircuit(q, c)

    # Hゲートを量子ビット0に適用して量子重ね合わせを作ります。
    qc.h(q[0])
    # CX (CNOT) ゲートを制御量子ビット0に目的量子ビット1にしておき
    # ベル状態の回路を作ります。
    qc.cx(q[0], q[1])
    # 観測ゲートで状態を観測します。
    qc.measure(q, c)

    # 使用可能なデバイスのリストを出します。.
    print("IBMQ backends: ", IBMQ.backends())

    # デバイス上に量子回路をコンパイルし実行します。
    backend_ibmq = IBMQ.get_backend('ibmqx4')
    job_ibmq = execute(qc, backend_ibmq)
    result_ibmq = job_ibmq.result()

    # 結果を表示します。
    print("real execution results: ", result_ibmq)
    print(result_ibmq.get_counts(qc))

IBM Qの認証のセットアップの仕方の詳細はインストール:ref:`qconfig-setup`のセクションを参考にしてください。


HPCオンラインのバックエンドの使用
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 ``ibmq_qasm_simulator_hpc`` オンラインバックエンドには、以下の構成可能なパラメーターがあります：

- ``multi_shot_optimization``: ブール値 (真偽値)
- ``omp_num_threads``: 1から16の間の整数

パラメータは、:func:`qiskit.compile`と:func:`qiskit.execute` を
``hpc`` パラメータで実行します。 例えば：

.. code-block:: python

    qiskit.compile(circuits,
                   backend=backend,
                   shots=shots,
                   seed=88,
                   hpc={
                       'multi_shot_optimization': True,
                       'omp_num_threads': 16
                   })

もし、``ibmq_qasm_simulator_hpc``バックエンドが使用され、``hpc`` パラメータが指定されていない場合、
以下のデフォルト値が使用されます：

.. code-block:: python

    hpc={
        'multi_shot_optimization': True,
        'omp_num_threads': 16
    }

これらのパラメータは、``ibmq_qasm_simulator_hpc``のために使われ、
別のバックエンドと一緒に使用される場合、Terraから警告を出されるとともに、
Noneにリセットされます。
