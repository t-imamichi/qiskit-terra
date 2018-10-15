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
.. _qconfig-setup:


3. APIトークンとQEの資格の設定
----------------------------

-  `IBM Q <https://quantumexperience.ng.bluemix.net>`__
   のアカウントがない場合は作成します。
-  IBM Q のウェブサイトの“My Account” > “Advanced”
   からAPIトークンを取得します。
 
3.1 資格を自動的にロードする
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Qiskit Terra 0.6以降、IBM Q量子デバイスにアクセスするための認証情報は、複数の場所から自動的にロードされるため、
IBM Q認証の設定が合理化されます。 インストール後にAPI資格情報を一度設定または保存すると、
次に使用する場合は、次のコマンドを実行するだけですみます：

.. code:: python

    from qiskit import IBMQ

    IBMQ.load_accounts()

この `` IBMQ.load_accounts（） ``の呼び出しは、（必要に応じて）複数のソースからの資格情報を自動的にロードし、
IBM Qを認証して、オンラインのデバイスをあなたのプログラムで利用できるようにします。 
自動認証を呼び出す前に、以下のいずれかの方法で資格情報を保存してください：

3.1.1 API 認証情報をローカルに保存する
"""""""""""""""""""""""""""""""""""

ほとんどのユーザーにとって、API認証情報をローカルに保存するのが最も便利な方法です。
あなたの情報は `qiskitrc`という設定ファイルでローカルに保存され、
一度保存されると、プログラムに明記しなくても、資格情報を使うことができます。

あなたの情報を保存するには、簡単に以下を実行します：

.. code:: python

    from qiskit import IBMQ

    IBMQ.save_account('MY_API_TOKEN')

あなたのトークンで MY_API_TOKEN の部分を置き換えてください。

あながた、IBM Q network を使用している場合は、
q-consoleのアカウントページにある `url`の文字と必要なその他の追加情報（プロキシ情報など）を
` IBMQ.save_account（） `に渡す必要があります。

.. code:: python

    from qiskit import IBMQ

    IBMQ.save_account('MY_API_TOKEN', url='https://...')

3.1.2 API 認証情報を環境変数からロードする
"""""""""""""""""""""""""""""""""""""""""""""""""""""

より高度なユーザーの場合は、環境変数からAPI資格情報をロードすることができます。 
具体的には、次の環境変数を設定できます:

* `QE_TOKEN`,
* `QE_URL`

使用環境にこれらが存在する場合、ディスクに保管されている認証情報よりも優先されます。


3.1.3 Qconfig.pyからAPI 認証情報をロードする
""""""""""""""""""""""""""""""""""""""""""

0.6より前のバージョンのQiskitで設定された設定との互換性のために、
プログラムが呼び出されるディレクトリにある `` Qconfig.py``ファイルに資格情報を保存することもできます。 
便宜上、リファレンスとして使えるように、このファイルのデフォルトバージョンを用意しています。
あなたの好きなエディタを使って、あなたのプログラムフォルダに以下の内容の `` Qconfig.py``ファイルを作成してください：


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

例えば、完全に設定された有効な `` Qconfig.py``ファイルは次のようになります：

.. code:: python

    APItoken = '123456789abc...'

    config = {
        'url': 'https://quantumexperience.ng.bluemix.net/api'
    }

IBM Qユーザーの場合、有効で完全に構成された `` Qconfig.py``ファイルは次のようになります：

.. code:: python

    APItoken = '123456789abc...'

    config = {
        'url': 'https://quantumexperience.ng.bluemix.net/api',
        # The following should only be needed for IBM Q users.
        'hub': 'MY_HUB',
        'group': 'MY_GROUP',
        'project': 'MY_PROJECT'
    }

`` Qconfig.py``ファイルがあなたのディレクトリに存在する場合、それは環境変数やディスクにローカルに保存された資格情報より優先されます。

3.2 認証情報を手動でロードする
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

もっと複雑なシナリオや、複数のアカウントをより細かく制御する必要のあるユーザーの場合は、
APIトークンとその他のパラメータを `` IBMQ.enable_account（） ``関数に
直接渡します。これは、自動的にロードされた認証情報を無視し、引数を直接使用しま。
例えば：

.. code:: python

    from qiskit import IBMQ

    IBMQ.enable_account('MY_API_TOKEN', url='https://my.url')

は、設定ファイル、環境変数、または `` Qconfig.py``ファイルなどに格納された設定にかかわらず、
`` MY_API_TOKEN``と指定されたURLを使って認証されます。

`` Qconfig.py``ファイルから手動でロードすることもできます：

.. code:: python

    from qiskit import IBMQ
    import Qconfig

    IBMQ.enable_account(Qconfig.APIToken, **Qconfig.config)


複数の資格情報を使用する方法の詳細については、 `` qiskit.IBMQ``のドキュメントを参照してください。


Jupyterを使ったチュートリアルのインストール
===============================

QISKitプロジェクトはチュートリアルをJupyterノートブックの形式で提供します。
ノートブックはPythonのコードが埋め込まれたウェブページのようなものです。
埋め込まれたコードを実行するには``Shift+Enter``を押すか、
ページ上部のツールバーを使います。
出力は即座にページの下に表示されます。多くの場合埋め込まれたコードは上から順に実行します。
チュートリアルを使いはじめるには以下の通りにします。


QISKitプロジェクトはチュートリアルをJupyterノートブックの形式で提供します。
JupyterノートブックはPythonコードの「セル」が埋め込まれたWebページです。 
詳細な手順は `チュートリアルレポジトリ` _を参照してください。


トラブルシューティング
===============

The installation steps described on this document assume familiarity with the
Python environment on your setup (for example, standard Python, ``virtualenv``
or Anaconda). Please consult the relevant documentation for instructions
tailored to your environment.

Depending on the system and setup, appending "sudo -H" before the
``pip install`` command could be needed:

.. code:: sh

    pip install -U --no-cache-dir qiskit



.. _tutorials: https://github.com/Qiskit/qiskit-tutorial
.. _tutorials repository: https://github.com/Qiskit/qiskit-tutorial
.. _documentation for contributors: https://github.com/Qiskit/qiskit-terra/blob/master/.github/CONTRIBUTING.rst
.. _Qconfig.py.default: https://github.com/Qiskit/qiskit-terra/blob/stable/Qconfig.py.default   




