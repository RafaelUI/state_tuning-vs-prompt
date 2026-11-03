#!/usr/bin/env python3
"""30.10: gen-run --distill --train top|lora (сильный контроль к обучению состояния) и --top-in / --lora-in в генерации."""
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, (n, old[:80])
    s = s.replace(old, new)


# шапка
rep("""//  Строки t.txt — «вид: вопрос»:""",
    """//  --train state|top|lora (30.10) — что учится на тех же ответах учителя:
//  state (умолчание) — начальное состояние; top — веса верхних слоёв
//  [--top-from K, умолчание nLayer−4] и ln_out целиком; lora — адаптеры
//  LoRA ранга --lora-rank (16) на tmix всех слоёв. Для top/lora — без
//  системной строки и без начального состояния, расписание как у состояния
//  (cosine, lrMin = lr/10, β₂ 0.99, отсечение 1.0); файл — --weights-out.
//  Генерация: --top-in f (верхние слои) / --lora-in f (адаптеры, сливаются
//  в веса; ранг — --lora-rank).
//  Строки t.txt — «вид: вопрос»:""")

# аргументы
rep("""    var sbCastAB = 0, sbLin = false, sbNoCast = false, sbArms = "off,wkv,lin"
}""", """    var sbCastAB = 0, sbLin = false, sbNoCast = false, sbArms = "off,wkv,lin"
    var trainWhat = "state", topFrom: Int? = nil, loraRank = 16, weightsOut = "", topIn: String? = nil, loraIn: String? = nil
}""")
rep("""        case "--sb-profile": a.sbProfile = true""",
    """        case "--train": a.trainWhat = v()                  // state | top | lora (30.10)
        case "--top-from": a.topFrom = Int(v())
        case "--lora-rank": a.loraRank = max(1, Int(v()) ?? 16)
        case "--weights-out": a.weightsOut = v()
        case "--top-in": a.topIn = v()
        case "--lora-in": a.loraIn = v()
        case "--sb-profile": a.sbProfile = true""")

# обучение весов вместо состояния — после того как пачки собраны
rep("""    var cfg = StateTuneConfig()
    cfg.lr = args.lr; cfg.lrMin = args.lr / 10;""", """    if args.trainWhat != "state" {
        // сильный контроль (30.10): те же данные и цикл, учатся веса; начального состояния нет
        guard ["top", "lora"].contains(args.trainWhat) else { print("--train: state | top | lora"); exit(2) }
        guard !args.weightsOut.isEmpty else { print("--train \\(args.trainWhat): нужен --weights-out"); exit(2) }
        guard heldBatches.isEmpty, args.evalStates.isEmpty else {
            print("--held-out / --eval-states с --train \\(args.trainWhat) не поддержаны"); exit(2)
        }
        guard args.initMode == "zero" else { print("--train \\(args.trainWhat): --init не применим (состояния нет)"); exit(2) }
        let lc = LoRAConfig(lr: args.lr, gradClip: args.gradClip ?? 1.0, beta1: 0.9, beta2: 0.99,
                            maxSteps: args.steps, warmupSteps: min(20, args.steps / 10), cacheLimitGB: 0.25,
                            logEvery: args.logEvery, schedule: .cosine, lrMin: args.lr / 10)
        let t1 = Date()
        let log: (Int, Float, Float, Double) -> Void = { step, loss, gn, peak in
            if step % args.logEvery == 0 || step == 1 {
                print(String(format: "  шаг %d: лосс %.4f, норма градиента %.3g, пик MLX %.0f МБ, %.0f с", step, loss, gn,
                             peak, Date().timeIntervalSince(t1)))
                fflush(stdout)
            }
        }
        let url = URL(fileURLWithPath: expand(args.weightsOut))
        if args.trainWhat == "top" {
            let from = args.topFrom ?? max(0, base.cfg.nLayer - 4)
            let r = TopLayerFinetune.run(base, from: from, nextBatch: nextBatch, config: lc, onStep: log)
            let note = "distill top from \\(from), \\(args.steps) шагов, lr \\(args.lr)"
            try TopLayerFinetune.save(base, result: r, to: url, note: note)
            print(String(format: "верхние слои сохранены: %@ (%@; весов %d), шаг %d: лосс %.4f, %.0f с", args.weightsOut,
                         note, r.keys.count, r.steps, r.finalLoss, Date().timeIntervalSince(t1)))
        } else {
            let info = LoRA.add(to: base, spec: LoRASpec(rank: args.loraRank, alpha: Float(args.loraRank)))
            let r = LoRAFinetune.run(base, nextBatch: nextBatch, config: lc, onStep: log)
            try LoRA.save(base, to: url)
            print(String(format: "адаптеры LoRA сохранены: %@ (r %d, обучаемых %d, %d шагов, lr %g), лосс %.4f, %.0f с",
                         args.weightsOut, args.loraRank, info.trainableParams, r.steps, args.lr, r.finalLoss,
                         Date().timeIntervalSince(t1)))
        }
        return
    }

    var cfg = StateTuneConfig()
    cfg.lr = args.lr; cfg.lrMin = args.lr / 10;""")

# генерация: подключить веса до состояния
rep("""if let path = args.stateIn {
    if path.hasSuffix(".pth") {
        let (state, ignored)""", """if let p = args.topIn {
    let keys = try TopLayerFinetune.load(base, from: URL(fileURLWithPath: expand(p)))
    print("подключены верхние слои: \\(keys.count) весов из \\(p)")
}
if let p = args.loraIn {
    _ = LoRA.add(to: base, spec: LoRASpec(rank: args.loraRank, alpha: Float(args.loraRank)))
    try LoRA.load(base, from: URL(fileURLWithPath: expand(p)))
    LoRA.merge(base)                                    // в веса: инференс идёт обычным путём
    print("подключены адаптеры LoRA r=\\(args.loraRank) из \\(p) (слиты в веса)")
}
if let path = args.stateIn {
    if path.hasSuffix(".pth") {
        let (state, ignored)""")
open(p, "w", encoding="utf-8").write(s)
print("ok")
