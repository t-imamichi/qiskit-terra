==========================
インストールとセットアップ
==========================

インストール
============

1. ツールの入手
---------------

QISKitを利用するには少なくとも `Python 3.5か以降 <https://www.python.org/downloads/>`__ と
`Jupyter Notebooks <https://jupyter.readthedocs.io/en/latest/install.html>`__ を
インストールしておく必要があります。
(後者はチュートリアルで対話的に操作することをお勧めします)。

一般ユーザーにQISKitが依存する多くのライブラリが含まれている
`Anaconda 3 <https://www.continuum.io/downloads>`__ という
Python ディストリビューションをお勧めします。

2. インストール
-------------------

QISKitをインストールする最も簡単な方法はPIP tool(Pythonのパッケージマネージャー)を利用することです。

.. code:: sh

    pip install qiskit

これですべての依存関係に沿った、最新の安定したリリースがインストールされます。

3. APIトークンとQEの資格の設定
----------------------------

-  `IBM Q <https://quantumexperience.ng.bluemix.net>`__
   のアカウントがない場合は作成します。
-  IBM Q のウェブサイトの“My Account” > “Advanced”
   からAPIトークンを取得します。
 
3.1 資格を自動的にロードする
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Qiskit Terra 0.6以降、IBM Q の量子デバイスにアクセスするための認証情報は、複数の場所から自動的にロードされるため、
IBM Q の認証設定が合理化されています。 インストール後にAPI資格情報を一度設定または保存すると、
次に使用する場合は、次のコマンドを実行するだけですみます：

.. code:: python

    from qiskit import IBMQ

    IBMQ.load_accounts()

この IBMQ.load_accounts（） の呼び出しは、（必要に応じて）複数のソースからの資格情報を自動的にロードし、
IBM Qを認証して、オンラインのデバイスをあなたのプログラムで利用できるようにします。 
自動認証を呼び出す前に、以下のいずれかの方法で資格情報を保存してください：

3.1.1 API 認証情報をローカルに保存する
"""""""""""""""""""""""""""""""""""

ほとんどのユーザーにとって、API認証情報をローカルに保存するのが最も便利な方法です。
あなたの情報は qiskitrc という設定ファイルでローカルに保存され、
一度保存されると、プログラムに明記しなくても、資格情報を使うことができます。

あなたの情報を保存するには、以下で簡単に実行できます：

.. code:: python

    from qiskit import IBMQ

    IBMQ.save_account('MY_API_TOKEN')

MY_API_TOKEN の部分をあなたのトークンに置き換えてください。

あながた、IBM Q network を使用している場合は、
q-consoleのアカウントページにある url の文字と必要なその他の追加情報（プロキシ情報など）を
 IBMQ.save_account（） に渡す必要があります。

.. code:: python

    from qiskit import IBMQ

    IBMQ.save_account('MY_API_TOKEN', url='https://...')

3.1.2 API 認証情報を環境変数からロードする
"""""""""""""""""""""""""""""""""""""""""""""""""""""

より高度なユーザーの場合は、環境変数からAPI資格情報をロードすることができます。 
具体的には、次の環境変数を設定できます:

* `QE_TOKEN`,
* `QE_URL`

ディスクに保管されている認証情報よりも、これらの環境変数が優先されます。


3.1.3 Qconfig.pyからAPI 認証情報をロードする
""""""""""""""""""""""""""""""""""""""""""

0.6より前のバージョンのQiskitでの設定との互換性のために、
プログラムが呼び出されるディレクトリにある Qconfig.py ファイルに資格情報を保存することもできます。 
リファレンスとして使えるように、このファイルのデフォルトバージョンを用意しています。
あなたの好きなエディタを使って、以下の内容の  Qconfig.py ファイルを作成して、あなたのプログラムフォルダに置いてください：


.. code:: python

    APItoken = 'PUT_YOUR_API_TOKEN_HERE'

    config = {
        'url': 'https://quantumexperience.ng.bluemix.net/api',

        # If you have access to IBM Q features, you also need to fill the "hub",
        # "group", and "project" details. Replace "None" on the lines below
        # with your details from Quantum Experience, quoting the strings, for
        # example: 'hub': 'my_hub'
        # You will also need to update the 'url' above, pointing it to your custom
        # URL for IBM Q.
        'hub': None,
        'group': None,
        'project': None
    }

    if 'APItoken' not in locals():
        raise Exception('Please set up your access token. See Qconfig.py.')

そして、以下の行を修正します：

* 最初の行(``APItoken = 'PUT_YOUR_API_TOKEN_HERE'``)の ' ' の間のスペースにAPIトークンをコピー/貼り付け。

* IBM Q の機能にアクセスできる場合は、url, hub, group, および projectの値も設定する必要があります。 
これを行うには、IBM Qのアカウント・ページにある値をconfig変数に入力します。

例えば、完全に設定された有効な Qconfig.py ファイルは次のようになります：

.. code:: python

    APItoken = '123456789abc...'

    config = {
        'url': 'https://quantumexperience.ng.bluemix.net/api'
    }

IBM Qユーザーの場合、有効で完全に構成された Qconfig.py ファイルは次のようになります：

.. code:: python

    APItoken = '123456789abc...'

    config = {
        'url': 'https://quantumexperience.ng.bluemix.net/api',
        # The following should only be needed for IBM Q users.
        'hub': 'MY_HUB',
        'group': 'MY_GROUP',
        'project': 'MY_PROJECT'
    }

Qconfig.py ファイルがあなたのディレクトリに存在する場合、それは環境変数やディスクにローカルに保存された資格情報より優先されます。

3.2 認証情報を手動でロードする
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

もっと複雑なシナリオや、複数のアカウントをより細かく制御する必要のあるユーザーの場合は、
APIトークンとその他のパラメータを  IBMQ.enable_account（） 関数に
直接渡します。これは、自動的にロードされた認証情報を無視し、この引数を直接使用します。
例えば：

.. code:: python

    from qiskit import IBMQ

    IBMQ.enable_account('MY_API_TOKEN', url='https://my.url')

は、設定ファイル、環境変数、または  Qconfig.py ファイルなどに格納された設定にかかわらず、
 MY_API_TOKEN と指定されたURLを使って認証されます。

 Qconfig.py ファイルから手動でロードすることもできます：

.. code:: python

    from qiskit import IBMQ
    import Qconfig

    IBMQ.enable_account(Qconfig.APIToken, **Qconfig.config)


複数の資格情報を使用する方法の詳細については、qiskit.IBMQ のドキュメントを参照してください。


Jupyterを使ったチュートリアルのインストール
===============================

QISKitプロジェクトはチュートリアルをJupyterノートブックの形式で提供します。
JupyterノートブックはPythonコードの「セル」が埋め込まれたWebページです。 
詳細な手順は　`tutorials repository`_ を参照してください。


トラブルシューティング
===============

このドキュメントで説明しているインストール手順は、Python環境（標準のPython、virtualenv、Anacondaなど）に
精通していることを前提としています。 ご使用の環境に合わせた手順については、該当するドキュメントを参照してください。

システムとセットアップによっては、pip installコマンドの前に "sudo -H"を追加する必要があります：

.. code:: sh

    pip install -U --no-cache-dir qiskit



.. _tutorials: https://github.com/Qiskit/qiskit-tutorial
.. _tutorials repository: https://github.com/Qiskit/qiskit-tutorial
.. _documentation for contributors: https://github.com/Qiskit/qiskit-terra/blob/master/.github/CONTRIBUTING.rst
.. _Qconfig.py.default: https://github.com/Qiskit/qiskit-terra/blob/stable/Qconfig.py.default   




