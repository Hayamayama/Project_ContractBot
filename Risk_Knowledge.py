# risk_knowledge.py
# 這是合約風險分類知識庫。
# 集中管理所有風險定義，方便未來擴充與維護。

CONTRACT_RISK_KNOWLEDGE_BASE = [
    # =================================================================
    # ==                            高風險條款                         ==
    # =================================================================
    {
        "risk_level": "High",
        "clause_type": "賠償責任 (Indemnification)",
        "example_scenario": "甲方應賠償並使乙方免受任何因本合約引起或與之相關的索賠、損失、責任，即使該損失是由乙方的部分或全部過失所造成。",
        "risk_analysis": "此為『單向無限賠償』且『包含對方過失』的條款。我方需為對方的錯誤買單，風險無限擴大，極不公平，可能帶來毀滅性的財務打擊。"
    },
    {
        "risk_level": "High",
        "clause_type": "智慧財產權 (Intellectual Property)",
        "example_scenario": "甲方在履行本合約期間為乙方所創造的所有智慧財產權，包括甲方在簽約前已擁有的背景技術，其所有權皆歸乙方所有。",
        "risk_analysis": "這會導致公司的核心資產（特別是簽約前就擁有的『背景技術』）被無償轉讓，嚴重削弱長期競爭力與未來發展。"
    },
    {
        "risk_level": "High",
        "clause_type": "數據使用權 (Data Usage Rights)",
        "example_scenario": "乙方有權永久、無償、不可撤銷地使用、複製、修改、再授權甲方提供的所有數據，並用於乙方任何商業目的。",
        "risk_analysis": "公司的數據是重要資產。此條款讓對方可以無限制地濫用我方數據，甚至分享給競爭對手，帶來無法預估的商業和法律風險。"
    },
    {
        "risk_level": "High",
        "clause_type": "合約自動續約 (Automatic Renewal)",
        "example_scenario": "本合約期滿後，若任一方未在到期日 90 天前以書面通知反對，本合約將自動續約五年，續約條件不變。",
        "risk_analysis": "若疏忽未通知，公司將被一個可能已不符效益的合約『惡意長期綁定』，且無法輕易退出，造成持續性的財務損失。"
    },
    {
        "risk_level": "High",
        "clause_type": "排他性/獨家性 (Exclusivity)",
        "example_scenario": "在合約期間及結束後三年內，甲方不得與任何從事與乙方相同或相似業務的第三方進行合作。",
        "risk_analysis": "過於寬泛且長期的排他性條款，會將公司鎖死，無法拓展其他市場或與更有利的夥伴合作，嚴重阻礙業務發展。"
    },

    # =================================================================
    # ==                            中風險條款                         ==
    # =================================================================
    {
        "risk_level": "Medium",
        "clause_type": "付款條件 (Payment Terms)",
        "example_scenario": "甲方應於收到乙方請款發票後 90 天內付清所有款項。",
        "risk_analysis": "過長的付款週期 (Net 90) 會嚴重影響收款方的現金流健康，雖然帳款能收到，但期間的資金壓力與機會成本很高。"
    },
    {
        "risk_level": "Medium",
        "clause_type": "工作範圍 (Scope of Work)",
        "example_scenario": "乙方應提供甲方所需的一切相關技術支援與顧問服務，以確保專案成功。",
        "risk_analysis": "定義模糊，『一切相關』可以被無限解釋，容易導致『範圍蠕變』(Scope Creep)，使我方需投入遠超預期的資源與人力，造成專案虧損。"
    },
    {
        "risk_level": "Medium",
        "clause_type": "服務水平協議 (SLA)",
        "example_scenario": "系統正常運行時間需達到 99.99%。若未達標，每下降 0.01%，乙方需賠償當月服務費的 5% 作為違約金。",
        "risk_analysis": "雖然設定了標準，但罰則可能過於嚴苛。一次技術問題就可能導致該月利潤損失大半，需要投入額外成本來確保達標。"
    },
    {
        "risk_level": "Medium",
        "clause_type": "責任上限 (Limitation of Liability)",
        "example_scenario": "任一方在本合約下的總賠償責任上限，為過去三個月的合約總金額。",
        "risk_analysis": "此賠償上限可能過低。若對方違約造成我方巨大損失（如商譽損失、關鍵數據遺失），這個上限將遠遠不足以彌補我方的實際損失。"
    },
    {
        "risk_level": "Medium",
        "clause_type": "禁止招攬員工 (Non-solicitation)",
        "example_scenario": "合約期間及結束後兩年內，甲方不得直接或間接雇用或招攬任何曾經是乙方員工或承包商的人員。",
        "risk_analysis": "範圍過於寬泛，『曾經是』可能導致我方無法雇用業界的優秀人才，只因他多年前曾在對方公司任職，不合理地限制了人才引進。"
    },

    # =================================================================
    # ==                            低風險條款                         ==
    # =================================================================
    {
        "risk_level": "Low",
        "clause_type": "管轄法律與法院 (Governing Law)",
        "example_scenario": "本合約之解釋與適用，應以中華民國法律為準據法。因本合約所生之爭議，雙方同意以臺灣臺北地方法院為第一審管轄法院。",
        "risk_analysis": "選擇一個中立、合理且對雙方都方便的法律和法院，是標準作法，能確保爭議解決的穩定性與可預測性。"
    },
    {
        "risk_level": "Low",
        "clause_type": "保密期限 (Confidentiality)",
        "example_scenario": "雙方應對本合約中的機密資訊保密，保密義務於合約終止後持續三年。",
        "risk_analysis": "3 到 5 年的保密期是多數商業合約的標準慣例，既能保護資訊，又不會造成永久性的不合理負擔。"
    },
]

def get_risk_rubric_string():
    """將知識庫格式化為一個清晰的字串，方便注入到 Prompt 中。"""
    rubric_parts = []
    for item in CONTRACT_RISK_KNOWLEDGE_BASE:
        part = (
            f"- **風險等級: {item['risk_level']}**\n"
            f"  - **條款類型**: {item['clause_type']}\n"
            f"  - **風險分析**: {item['risk_analysis']}\n"
        )
        rubric_parts.append(part)
    return "\n".join(rubric_parts)