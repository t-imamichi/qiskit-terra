QISKit入門
==========

:class:`~qiskit.QuantumCircuit` がコードを書く際の起点になります。
回路 (またはIBM Q Experienceを使ったことがある人はスコア）は、 :class:`~qiskit.ClassicalRegister` objects
と:class:`~qiskit.QuantumRegister` objects と:mod:`gates <qiskit.extensions.standard>`
で構成されます。:ref:`top-level functions <qiskit_top_level_functions>`を通じて、
回路をリモートの量子デバイスまたはローカルシミュレータのバックエンドに送信し、
結果を収集して、さらに分析します。

シミュレーター上で量子回路を設計して実行するには、以下のようにします。

.. code-block:: python

    # Import the Qiskit SDK
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit import execute, Aer

    # Create a Quantum Register with 2 qubits.
    q = QuantumRegister(2)
    # Create a Classical Register with 2 bits.
    c = ClassicalRegister(2)
    # Create a Quantum Circuit
    qc = QuantumCircuit(q, c)

    # Add a H gate on qubit 0, putting this qubit in superposition.
    qc.h(q[0])
    # Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
    # the qubits in a Bell state.
    qc.cx(q[0], q[1])
    # Add a Measure gate to see the state.
    qc.measure(q, c)

    # See a list of available local simulators
    print("Aer backends: ", Aer.backends())

    # Compile and run the Quantum circuit on a simulator backend
    backend_sim = Aer.get_backend('qasm_simulator')
    job_sim = execute(qc, backend_sim)
    result_sim = job_sim.result()

    # Show the results
    print("simulation: ", result_sim )
    print(result_sim.get_counts(qc))

:func:`~qiskit.Result.get_counts` メソッドが``state:counts``のペアの辞書オブジェクトを出力します。

.. code-block:: python

    {'00': 531, '11': 493}

量子プロセッサー
----------------

ユーザーはIBM Q Experience (QX)のクラウドプラットホームを通じて、
実機の量子コンピューター（量子プロセッサー）で回路を実行することができます。
現在QXでは、以下のチップが利用可能です:

-   ``ibmqx4``: `5-qubit backend <https://ibm.biz/qiskit-ibmqx4>`_

-   ``ibmq_16_melbourne``: `16-qubit backend <https://github.com/Qiskit/qiskit-backend-information/tree/master/backends/melbourne/V1>`_

最新の実機の詳細と現在使用可能かどうかについては
`IBM Q experience backend information <https://github.com/Qiskit/ibmqx-backend-information>`_
と `IBM Q experience devices page <https://quantumexperience.ng.bluemix.net/qx/devices>`_　を参照してください。

.. include:: example_real_backend.rst

プロジェクト構成
----------------

Pythonのプログラム例は *examples* ディレクトリに、
テストスクリプトは *test* ディレクトリにあります。
*qiskit* ディレクトリがTerraのメインモジュールです。
