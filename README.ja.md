# CUELOOP

![tests](https://github.com/Unjuno/cueloop/actions/workflows/test.yml/badge.svg)
![license](https://img.shields.io/badge/license-MIT-blue.svg)

> 英語版は [README.md](README.md) を参照してください。この文書はその日本語版です。

CUELOOP は、話し手が話している**最中に**、話し手が次に言おうとしている短い区間を予測し、
次の節が生成される前に「cue（促し）」を挿入するシステムです。既定では cue は**ステム**
（操作を体現する、その場面にもっとも妥当な主述完結文、例:「根拠は、
観測データでは弱いからです。」）で、話し手はそれを採用・修正できます。
`--cue-style bare` で従来の短い操作語（「根拠は？」）に戻せます。

検証しているのは弱い言語フレーミング仮説です：話し手が操作を発表する瞬間に
特定の操作語を聞くと次に話す内容が変わるか、そして繰り返しの練習は
推論の構造を改善するか。メカニズム自体は今でも測れます。ただし
「書く力・考える力の持続的改善」を主張するには、遅延かつ無支援の課題での検証が必要です。

## 現状

| 項目 | 状態 |
| --- | --- |
| 同期コア（予測→同期→判定→発話） | 動作。テスト127件、CIグリーン |
| macOS マイクアダプタ + ライブデモ | このMacでは動作（環境音依存） |
| 既定の音声 | 日本語・macOS `say -v Kyoko` |
| 合成データによる較正 | エンドツーエンドで効果を回収（docs/benchmark.md） |
| 人間のセッション | **まだなし** — H1/H2 の結果は存在しない |
| ライセンス | MIT |

即時の uptake はメカニズムの確認にすぎず、持続的改善の証拠ではありません。
人間のデータに基づく主張はまだ何もありません。ローカルにある日本語実話サンプルに対する
限定的な ASR 較正は `docs/real-speech.md` を参照。

## ステムcue（既定）とベアcue

既定の `--cue-style stem` は、疑問句ではなく完結文を話します。ベアcueは
欠けている操作の「名前」を言うだけですが、ステムcueはそれを「実演」します。
話し手は節の形と、操作と内容のつながりを耳で聞きます：

```text
bare: "根拠は？"                              （8文字、旧既定）
stem: "根拠は、観測データでは弱いからです。"   （この文脈用にLLMが生成）
```

バックエンドごとの生成元：

- `--decision local` / `--decision steer` — 判定呼び出しと同じJSON契約の中で
  ステムを書きます。追加の往復はありません（ステム時は `--max-tokens` が
  192以上に引き上げられ、JSON欠落を防ぎます）。
- `--decision rule` / `--decision linear` — `--async-decision`（既定有効）では、
  部分的な転写が来るたびにステム生成がバックグラウンドで先行します。境界では
  プリコンピュート済みのステムが境界＋挿入遅延に乗って届きます。間に合わない場合は、
  遅れたステムではなく**時間内に届くベアcue**が届きます（`--decision-fallback none`
  なら待ちますが、節を逃すことがあります）。`--no-async-decision` は同期経路を強制し、
  境界で数秒ブロックして節を逃す可能性が高くなります。
  いずれの場合も `--stem-timeout`（既定1.5秒）で制限され、失敗は
  予測スパン埋め込みのテンプレート（`根拠は、{span}からです。`）へ、それも無理なら
  ベアcueへフォールバックします。ローカルサーバーが起動時に到達できない場合は
  警告を出してテンプレート運用に切り替わります。
- `--assignment random-operation` — ラッパーが操作を（一様に）引き、その**引かれた**
  操作のステムを描画します。

ステムは長い（既定上限 `--max-stem-characters` 48）ので、読み上げに1秒未満ではなく
数秒かかり、TTSのエコー抑制も文字数に応じて延長されます（`1000 + 150ms`/文字、上限8秒）。
実験では `NEUTRAL` をステムに長さを合わせるか、`--cue-style bare` を使ってください。
`intervention` ログには `cue_style` とステム全文が記録されます。

## ループ

```text
音声イベント（部分／確定／VAD）
        ↓
節境界または操作の発表（「結論は」「つまり」、休止）
        ↓
次の短い区間を予測
        ↓
同期: 発話を予測と照合し、発散を推定
        ↓
ローカル判定モデル: 次の区間に役立つ操作を省略していないか？
        ↓
cue を割り当てる（実験変数）
        ↓
次の節が生成される前に cue を話す
        ↓
継続を記録 → uptake・carryover／予測一致率
```

2つの関心は意図的に分離されています：

| 関心 | 担当 | 実験上の役割 |
| --- | --- | --- |
| いつ介入するか | `RuleDecisionModel` / `LinearDecisionModel` / `LocalLLMDecisionModel` | 条件間で同じに保つ |
| どの cue が聞こえるか | `CueAssigner` | 操作変数（canonical, fixed, alternate, randomized） |

モデルが実測効果から cue を選んではいけません。フレーミング効果と選択バイアスが
交絡します。効果はオフラインでランキングできて、実験ではアサインするだけです。

## クイックスタート

```bash
python -m pip install -e .

# 決定的リプレイ: 900ms の境界で cue を 100ms 後に届けます
cueloop-sync --events experiments/sync-smoke.jsonl \
  --log artifacts/sync-smoke.jsonl \
  --report artifacts/sync-report.json
```

ローカル判定モデルは OpenAI 互換のサーバー上で動きます。既定のサーバーは
Ollama（`http://127.0.0.1:11434`）で、モデルはサーバーの一覧から選ばれます。

```bash
cueloop-sync --events experiments/sync-smoke.jsonl \
  --decision local
```

ローカルLLMは意味論的な事前知識にはなりますが、毎回の境界に使うには遅すぎます。
同じログから速い線形ポリシーをオフラインで学習できます：

```bash
cueloop-sync --events experiments/sync-smoke.jsonl --decision local \
  --log logs/session-001.jsonl
scripts/train_policy.py logs/*.jsonl \
  --label carryover --output artifacts/sync-policy.json
cueloop-sync --events experiments/sync-smoke.jsonl --decision linear
```

linear バックエンドは rule が「欠けている」と判断した操作だけをスコアします。
閾値付きの内積なので、境界に LLM のレイテンシは加わりません。

実験条件はすべてのログに記録してください：

```bash
cueloop-sync --events experiments/sync-smoke.jsonl \
  --decision local \
  --assignment random --seed 7 \
  --condition-label CUE
```

## デモ

決定的リプレイを実時間で（約3秒）：

```bash
scripts/demo.sh replay
```

スクリプトはローカルの `.venv` のエントリポイントがあればそれを使い、macOS `say` で
cue を読み上げ、`artifacts/demo/` にログを書きます。`CUELOOP_TTS=0` でミュート、
`CUELOOP_DECISION=local` でローカルモデル＋ruleフォールバックを使います。

ライブマイクのデモ（`結論は`／`つまり`／`前提は`／`逆に` を含む文を話してください）：

```bash
scripts/demo.sh live 45
```

`.build/macos-speech-events` が必要です。ビルド方法は次の節を見てください。
`scripts/demo.sh live 10` が簡易チェックです。

`cueloop-sync --realtime --events <file>` はイベントのタイムスタンプでペースを取ります。
なしの場合、リプレイは可能な限り速く回ります。

## macOS ライブセッション

マイクアダプタを一度だけビルドします：

```bash
mkdir -p .build
swiftc tools/macos_speech_events.swift -o .build/macos-speech-events \
  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist \
  -Xlinker tools/macos_speech_events.Info.plist
```

ループを実行します。オプションのようなトークンを含む引数は JSON 形式で：

```bash
cueloop-sync \
  --event-command-json '[".build/macos-speech-events","--locale","ja-JP"]' \
  --decision local \
  --tts-command-json '["say","-v","Kyoko"]' \
  --condition-label CUE
```

`say -v Kyoko` のようにオプション風のトークンを含むコマンドは `--tts-command-json`、
`--tts-command say` のような単語1つなら普通の形で渡せます。

初回は macOS が音声認識とマイクの許可を求めます。話し手が空気伝導に調整された `say`
ボイスの場合、マイクは cue も聞きます。再生後の抑制窓が cue の語をマスクしますが、
*無生物の話し手* は cue に反応できないので uptake は構造的に0で、介入回数は SILENCE
アームとずれ得ます。そのような弱信号ループでは
`--adapter-args-json '["--speech-db","-50"]'` でアダプタの検出閾値を下げられます。
直接マイクに話す場合は不要です。

アダプタはリプレイと同じイベント契約を出力します：

```json
{"type":"speech_started","at_ms":0}
{"type":"transcript_partial","at_ms":900,"text":"この設計の結論は"}
{"type":"tick","at_ms":1200}
{"type":"transcript_final","at_ms":1600,"text":"この設計の結論は、観測データでは弱い。"}
{"type":"speech_ended","at_ms":1800}
{"type":"session_ended","at_ms":2000}
```

## 予測と判定バックエンド

- `--predictor ngram`（既定）— 小さな日本語コーパスで学習した文字 n-gram。短い休止に間に合う速さです。
- `--predictor local` — OpenAI 互換モデルが次の区間を予測。バックグラウンドで動き、
  n-gram にフォールバックします（`--async-prediction`）。遅い・不正なモデルが
  境界をブロックしません。小さなモデルはJSON契約にたまに失敗します。ログにフォールバック元が出ます。
- `--decision rule`（既定）— 予測する次の区間が操作を欠いていたら挿入します。
  直近の発話に隣接する操作を優遇します（`CONTEXT_LINKS` の手作り隣接事前知識）。
  節・直近の履歴・予測のどこにも操作信号がない場合は沈黙が正解なので黙ります。
  `--no-context-fit` で旧来の静的 PRIORITY 順に戻します。探索アーム
  （`--assignment random-operation`）は、rule が発火した後に欠けている集合から一様に引き続けます。
- `--decision linear` — `intervention`／`outcome`／`carryover` 記録からオフライン学習した
  小さな線形モデルで欠けている操作をスコアします。`--policy-model` 既定は
  `artifacts/sync-policy.json`。報酬は `carryover` が既定、`uptake` と `blend` もあります。
  閾値を超える carryover が予測されなければ runtime は沈黙します。文脈付き報酬モデルであり、
  完全な RL ではありません — 因果の主張には依然として操作のランダム化と探索が必要です。
- `--early-lead-ms`（既定0）— 休止の閾値に達する前に、現在の予測を使って cue を計画します。
  話し手が先に再開するかもしれない予測先読みなので、リードは小さく保ち、キャンセルは有効に。
- `--decision local` — モデルは欠けている操作だけを見て `CUE` か `SILENCE` を返します。
  不正な JSON、低い確信度、既に現れた操作、中身のない予測はすべて沈黙に倒れます。
  `--async-decision`（既定）で境界の前にプリコンピュートします。間に合わない場合は
  `--decision-fallback rule`（既定）が時間内に届け、`--decision-fallback none` はモデルを待ちます。
- `--decision steer`（実験的）— 軌道ステアリング。ローカルと同じJSON契約とフェイルクローズの
  規則ですが、次のチェックリスト項目ではなく、推論軌道を最も有用な方向へ動かす1手
  （反復している手を崩す、逆立した前提を試す、根拠のないジャンプに足場を要求する）を
  尋ねます。`rule`／`local` と同じペアプロトコルで比較するためのもので、それ自体を
  信じるためのものではありません。

安定して低レイテンシな実用構成は n-gram 予測＋速い判定バックエンドです。
M系ラップトップ上のローカル1.5B判定モデルは、非同期プリコンピュートでも
0.2〜0.6秒を要します — だから linear が存在します。`--insertion-delay-ms` 既定は100ms。

固定の配送レイテンシ目標はありません。スモークのフィクスチャは900msに境界があり、
挿入遅延が100msなので1000msに届きます — それはフィクスチャの性質であって目標ではありません。
意味があるのは発表された操作に対する相対時間です：cue は境界で始まり、次節が生成される前に
届くべきです。配送ログは `boundary_at_ms` と `boundary_to_delivery_ms` を記録するので、
（丸い数に最適化するのではなく）転写稿（`text_at_delivery`）と照合して時間を検証できます。

実時間挿入が存在する理由は、仮説が「先読み」型だからです。外から与えられた cue は、
これから生成される節をフレームしなければなりません。後からの要約や事後的提案は別の
メカニズムを検証することになります。実験が問うているのは、話し手が生成しなかった操作を
聞くことが次の節を変えるか、そしてその外部供給された操作が、cue なしの後の発話や
遅延の再構成へ持ち越されるか、です。

## 出力と測定

`--log` は追記専用のJSONLです：

- `prediction` — スパン・操作・確信度・出所・発話速度・レイテンシ
- `intervention` は `prediction_penalty_ms` と `prediction_precomputed` を含み、
  非同期予測のレイテンシを配送予算と混同しないようにします
- `intervention` — 要求操作・cue・アサイン・スケジュール・モデルの理由・
  `decision_source`（`precomputed`/`fallback`/`synchronous`）・`precomputed`
- `delivery` — `DELIVERED`／`CANCELLED`／`EXPIRED`
- `outcome` — 継続窓・cue の uptake・予測一致率
- `carryover` — cue なしの後のテキストに cue 操作が現れるか。セッション終了時に出力

`--report` は配送数・キャンセル数・操作ごとの uptake を集計します。`summary` ブロックは
予測一致率、`novel_uptake`（cue 前には無かったアサイン操作が継続に現れた回数）、
`carryover`（後の cue なし発話に現れる）、および `novel_carryover`（novel な場合の
carryover）を報告します。`repertoire_*` は最初の cue の前・セッション終了時・
その差分で観測された操作を示します。

即時 uptake はメカニズムの確認にすぎません。書く力・考える力の持続的改善の証拠には
なりません。carryover はより強い信号ですが、それでも証明ではありません：ブラインドな
ルーブリックで採点された遅延・無支援の再構成が、主張を支えられる結果です。

## ポリシー比較実験

どのバックエンドでも同じ予測器・境界検出器・アサイン方針・タイミングを使います。
変えるのは判定バックエンドだけです。実験ランナーはランダムな cue 文言でセッションを記録し、
マニフェストを書きます：

```bash
scripts/run_experiment.py --mode live --condition CUE \
  --decision rule --duration 60 --seed 1 --session rule-01
scripts/run_experiment.py --mode live --condition CUE \
  --decision linear --duration 60 --seed 1 --session linear-01
scripts/run_experiment.py --mode live --condition CUE \
  --decision local --duration 60 --seed 1 --session llm-01
```

`--condition SILENCE` は同じタイミング判断を記録しつつ cue を話しません。
`linear` は `--policy-model`（既定 `artifacts/sync-policy.json`）を使います。
ランナーは既定で `--cue-style bare` を渡し、過去セッションとの比較可能性を保ちます。
ステム条件を回したければ `--cue-style stem` を渡してください。
スモークテストには `--mode replay` があり、既定で TTS オフ・ペースなしなのでペアリプレイは
決定的です。`--realtime` を渡すとイベントのタイムスタンプでペースを取ります。

### 探索アーム：操作をランダム化する

`--assignment random` は文言だけをランダム化し、操作はポリシーが選びます。操作レベルの
因果比較には使えません。操作同士を直接比較したい場合は探索アームを使ってください：

```bash
scripts/run_experiment.py --mode replay --assignment random-operation \
  --condition CUE --decision rule --seed 1 \
  --events experiments/sync-carryover-smoke.jsonl --outcome-window-ms 1000 \
  --output-dir logs/experiment-explore-01 --session rule-cue-1
```

ラップされたポリシーは「いつ発火するか」だけを決めます。発火時に、それが検討したのと
同じ欠けている集合から一様に1つの操作が引かれるので、`requested_operation`（ポリシーが
望んだもの）と `operation`（話者が聞いたもの）が異なります。引きは `--seed` から再現可能で、
アサイン文字列に seed と文言戦略が記録されます。

### SILENCE 対照とペア解析

同じフィクスチャ・seed・タイミングで `--condition SILENCE` を回します。cue は保留されます：
レポートは `delivered=0` と `held=1` を示し、判断自体は記録されるので、タイミングと
操作の引きはペアで保たれます。ペアの実行は `--pair-key` を共有します（既定は
`<decision>-seed<seed>-<fixture>`）。解析はこれを使います：

```bash
scripts/analyze_experiment.py logs/experiment-explore-01/reports \
  --json artifacts/experiment-analysis.json \
  --markdown artifacts/experiment-analysis.md
```

セッションはペアでクラスタリングされます。ツールは条件ごとの集計と、
ペア差（CUE − SILENCE）をシード付きパーセンタイル・ブートストラップ（`--resamples`
既定2000、`--seed`、`--confidence` 既定0.95）で報告します。完全なペアが2組以上あるときだけ
区間が報告され、それ未満では `ci_low`／`ci_high` は `"insufficient"` になります。
CUE/SILENCE ペアが1組もないポリシーは、ゼロ効果ではなく `"not_comparable"` として報告されます。

リプレイの差分は構成上ゼロです。転写は固定ファイルなので、cue が次に話す内容を変えることは
原理的にありません。リプレイのゼロ差分はパイプラインとペアリングの動作を示すだけで、
cue が効かないことを示すものではありません。

### 合成データによる較正（これは人間の証拠ではない）

解析が既知の効果を回収できることを示すために、プログラムされた cue 効果を持つ生成的話者
モデルからペアレポートを生成できます：

```bash
scripts/synthesize_experiment.py --output-dir logs/synthetic-effect030 \
  --sessions 24 --clauses 8 --seed 0 --uptake-effect 0.30
scripts/analyze_experiment.py logs/synthetic-effect030 \
  --json artifacts/synthetic-effect030.json
```

完全な較正ベンチマーク（効果 × seed × バックエンドの解析グリッドと、
人為的な報酬構造を仕込んだ合成ログでのポリシー学習）はワンコマンドで再生成できます：

```bash
scripts/run_synthetic_benchmark.sh            # artifacts/benchmark/ を出力
```

現在の結果：ヌル9セルはすべて正確にゼロ、効果≥0.10の27セルはすべて CI が 0 を含みません。
エンドツーエンドのランタイム・シミュレーションではヌルは正確に +0.000 で、
0.2/0.4 の効果を CI が 0 を除外して回収します。
線形ポリシーの ROC-AUC は 0.695（学習時ホールドアウト）／0.664（独立ホールドアウト）、
天井 ≒ 0.73。`docs/benchmark.md` を参照。

ペアの CUE/SILENCE は、欠けている操作の経路と境界ごとの一様乱数（Common Random Numbers）を
共有し、プログラムした効果だけが異なります。`--uptake-effect 0.30`・24ペアで、解析は
rule/linear/llm 全体にわたって uptake +0.229〜+0.328（CI は 0 を含みません）を報告し、
`--uptake-effect 0.0` では正確に +0.000 を報告します。遅延差分は構成上 +0.000 です。
これらはパイプラインの較正結果であり、人間の発話に関する証拠ではありません。
すべてのパラメータは出力マニフェストに記録されます。

ポリシーファミリーでレポートを集計するには：

```bash
scripts/compare_policies.py logs/experiment/reports \
  --json artifacts/policy-comparison.json \
  --markdown artifacts/policy-comparison.md
```

`rule`／`linear`／`llm` ごとに、配送数・uptake・novel uptake・carryover・
novel carryover・repertoire growth・境界→配送遅延の中央値を報告します。
線形モデルは記録された介入から学習します：

```bash
scripts/train_policy.py logs/*.jsonl \
  --label carryover --output artifacts/sync-policy.json \
  --report artifacts/sync-policy-eval.json
scripts/evaluate_policy.py logs/*.jsonl \
  --model artifacts/sync-policy.json \
  --report artifacts/sync-policy-holdout.json
```

`train_policy.py` は層別のサンプルをホールドアウト（既定 `--eval-fraction 0.25`）し、
学習分割とホールドアウト分割の両方について accuracy・precision・recall・F1・ROC-AUC・
Brier score を書き出します。`evaluate_policy.py` は保存済みモデルを任意の後続ログで
採点します — 実運用の前に新しいセッションでポリシーを点検する方法です。
すべての指標はログに残った行動についてのものであり、デプロイ済みポリシーの
off-policy 推定ではありません。

即時 uptake はメカニズムの確認、carryover と遅延再構成は持続の結果です。
`docs/experiment-protocol.md` と `docs/sync-core.md` も参照してください。

## 失敗時の動作

- ステム生成の失敗は予測スパンのテンプレート → ベアcue へフォールバックします。
  ステムライターのつまずきがセッションを沈黙させることはありません。
  フェイルクローズするのは判定ではなく沈黙です。
- 予測器または判定モデルの失敗は cue を返しません。runtime は沈黙します。
- cue が節の前に話し手が予測から発散した場合・発話終了の場合・参加者が介入を
  一時停止した場合はキャンセルされます。
- 配送窓内に届かなかった cue は `EXPIRED` になります。
- TTS のエコーは、測定値（非同期モードでは推定値）のキューの長さ＋250ms 抑制されます。
- 話者コマンドの失敗は `DELIVERED` ではなく `SUPPRESSED` として記録されます。
- コンテキストは上限付きです。長いセッションでもプロンプトは無限には増えません。

## リポジトリ構成

```text
src/cueloop/sync/       イベントループ・予測器・ポリシー・ランタイム・CLI
scripts/demo.sh         リプレイとライブマイク用デモ
scripts/run_experiment.py ライブ／リプレイのポリシー比較セッション
scripts/synthesize_logs.py ポリシー学習を大規模化する合成セッションログ
scripts/analyze_power.py ペア CUE/SILENCE 設計の検出力解析（サンプルサイズ計画）
scripts/simulate_sessions.py エンドツーエンド・ランタイム・シミュレーション（合成日本語話者のペア CUE/SILENCE）
scripts/simulate_lux_replay.py 実話音声 replay（Lux corpus 328ファイル、ペアレポートを出力）
scripts/run_synthetic_benchmark.sh 較正ベンチマークの再生成（docs/benchmark.md）
scripts/train_policy.py 線形ポリシーのオフライン学習＋ホールドアウト指標
scripts/evaluate_policy.py 保存済みポリシーをログで採点
scripts/compare_policies.py rule/linear/LLM のレポート集計
scripts/analyze_experiment.py ブートストラップCI付きペア CUE/SILENCE 解析
tools/macos_speech_events.swift  macOS ASR アダプタ
tests/                  決定的なユニット／リプレイテスト
experiments/            リプレイのフィクスチャ
docs/                   設計・ベンチマーク・実話較正
CONTRIBUTING.md         開発と研究のグランドルール
```

以前のターン制スタックは削除済みです。sync コアが主経路で、イベント契約と macOS
アダプタは保持されています。
