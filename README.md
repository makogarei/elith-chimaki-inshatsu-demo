# LP行動ログアナライザー

Streamlit で動作する LP 行動ログの解析・可視化・改善提案ツールです。サンプルデータで動作確認しつつ、OpenAI GPT-5 系モデルを用いて詳細レポート（PDF任意）を生成できます。

## セットアップ
1. 依存関係をインストール
   ```bash
   pip install -r requirements.txt
   ```
   ※ PDF 生成には `weasyprint` が必要です。未導入の場合はインフォ表示のみとなります。

2. OpenAI API キーを設定  
   `.streamlit/secrets.toml` に以下を記載してください。
   ```toml
   openai_api_key = "sk-xxxx"
   ```

## 起動
```bash
streamlit run app.py
```

## 使い方（主要機能）
- **フィルター**: 期間・ページパス・訪問数下限・上位件数を指定して指標とグラフを再計算。
- **概要/行動/流入タブ**: 指標カード、トレンド、CTA散布図、流入元バー、上位/低位ページのテーブルを閲覧。
- **レポート生成**: 改善案タブでモデル名を指定し、A4 1ページ想定の詳細レポートを生成。`weasyprint` があれば PDF ダウンロード可。
- **LPプレビュー**: iframe で LP をプレビュー（CSP/X-Frame-Options により表示不可の場合あり）。

## データ
`datas/` 配下にサンプル CSV を同梱しています（トラフィック・流入・ボタンクリック）。未アップロード時はこれらを自動読み込みます。

## 動作確認
`python -m py_compile app.py` で構文チェック済み。Streamlit 実行で UI をご確認ください。
