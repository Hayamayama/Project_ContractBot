# Risk_Knowledge.py
# 這是合約風險分類知識庫。
# 集中管理所有風險定義，方便未來擴充與維護。
# 版本：3.0 (專家擴充版)

CONTRACT_RISK_KNOWLEDGE_BASE = [
    # =================================================================
    # ==                 🚨 高風險條款 (HIGH RISK)                    ==
    # ==   *通常需要法務強烈介入，不建議業務人員自行接受* ==
    # =================================================================
    {
        "risk_level": "High",
        "clause_type": "無限或單向賠償 (Unlimited or Unilateral Indemnification)",
        "keywords": [
            "indemnify", "hold harmless", "defend", "damages", "losses", "liabilities", "claims", "costs", "expenses",
            "unlimited liability", "caused by", "resulting from", "arising out of", "negligence of the indemnified party",
            "賠償", "使免受損害", "辯護", "損害", "損失", "責任", "索賠", "費用", "開支", "無限責任", "所導致",
            "對方的疏忽", "即使"
        ],
        "risk_analysis": "此為合約中最危險的條款之一，俗稱『空白支票』。它要求我方為對方，甚至是對方自己的錯誤或疏忽，承擔無限的財務賠償責任。這可能因單一事件就導致公司面臨毀滅性的財務打擊，必須嚴格拒絕或加上責任上限。"
    },
    {
        "risk_level": "High",
        "clause_type": "核心智慧財產權的轉讓 (Assignment of Core / Background IP)",
        "keywords": [
            "intellectual property", "IP", "assigns", "transfers", "ownership", "vests in", "background IP",
            "pre-existing technology", "all rights, title, and interest", "work product",
            "智慧財產權", "智財", "所有權", "轉讓", "歸屬", "背景技術", "既有技術", "所有權利", "工作成果"
        ],
        "risk_analysis": "此條款極具侵略性，它不僅轉讓為專案創造的新智財，更意圖剝奪我方在合作前就擁有的核心資產（背景技術）。接受此條款等於將公司的技術根基無償贈予對方，將嚴重削弱公司的長期競爭力與未來發展。"
    },
    {
        "risk_level": "High",
        "clause_type": "數據的無限與永久使用權 (Unrestricted & Perpetual Data Rights)",
        "keywords": [
            "data", "content", "perpetual", "irrevocable", "royalty-free", "sublicense", "distribute",
            "commercial purpose", "derivative works", "any purpose whatsoever",
            "數據", "資料", "內容", "永久", "不可撤銷", "免權利金", "再授權", "散布", "商業目的", "衍生作品", "任何目的"
        ],
        "risk_analysis": "此條款給予對方無限制的權利，可以永久地、無償地濫用我方提供的數據，甚至可能將其出售或提供給我們的競爭對手。在數據即資產的時代，這將引發不可預估的商業、法律及信譽風險。"
    },
    {
        "risk_level": "High",
        "clause_type": "對方單方面的隨意終止權 (Unilateral Termination for Convenience)",
        "keywords": [
            "terminate", "for convenience", "at any time", "in its sole discretion", "without cause", "for any reason",
            "終止合約", "隨時", "無需理由", "自行決定", "任何原因", "提前通知"
        ],
        "risk_analysis": "權力極度不對等的條款。它允許對方在我方投入大量時間、金錢與人力資源後，可以隨時、無理由地抽身離去，讓我方所有前期投入血本無歸。這使得合作關係極不穩定，我方將承擔所有專案沉沒成本的風險。"
    },
    {
        "risk_level": "High",
        "clause_type": "寬泛且長期的不競爭/排他性 (Broad & Long-term Non-Compete / Exclusivity)",
        "keywords": [
            "exclusive", "non-compete", "non-competition", "refrain from", "similar business", "affiliates",
            "排他", "獨家", "不競爭", "禁止", "避免", "相似業務", "關係企業"
        ],
        "risk_analysis": "過於寬泛（例如，地理範圍、業務定義模糊、包含所有關係企業）且過長（例如，合約終止後超過2年）的排他性條款，會將公司的發展道路完全鎖死，無法拓展新市場或與更有利的夥伴合作，嚴重阻礙公司的核心業務發展。"
    },
    {
        "risk_level": "High",
        "clause_type": "無上限的違約賠償金 (Uncapped Liquidated Damages)",
        "keywords": [
            "liquidated damages", "penalty", "service credits", "uncapped", "without limitation",
            "違約金", "罰款", "服務抵扣金", "無上限", "沒有限制"
        ],
        "risk_analysis": "雖然約定違約金是常見的，但若沒有設定一個合理的總上限（例如，不超過前12個月的服務費總額），可能會因為一次服務不穩定或交付延遲，導致賠償金額遠遠超過合約本身的價值，形成不成比例的巨大財務風險。"
    },

    # =================================================================
    # ==                ⚠️ 中風險條款 (MEDIUM RISK)                   ==
    # ==   *需要謹慎評估商業條件，並嘗試協商修改的條款* ==
    # =================================================================
    {
        "risk_level": "Medium",
        "clause_type": "過長的付款週期 (Extended Payment Terms)",
        "keywords": [
            "payment", "net 60", "net 90", "net 120", "invoice", "payment terms", "days after receipt",
            "付款", "帳期", "發票", "天內支付", "收到後"
        ],
        "risk_analysis": "帳期過長 (例如 Net 60 或 Net 90) 會嚴重影響我方的現金流健康。雖然帳款最終能收到，但在等待期間，我方需承受資金壓力並可能錯失其他投資機會，增加了營運的機會成本與財務風險。"
    },
    {
        "risk_level": "Medium",
        "clause_type": "模糊的工作範圍與驗收標準 (Vague SOW & Acceptance Criteria)",
        "keywords": [
            "scope of work", "SOW", "acceptance criteria", "reasonable efforts", "as needed", "to the satisfaction of",
            "工作範圍", "交付項目", "驗收標準", "合理努力", "視需要", "令其滿意為止"
        ],
        "risk_analysis": "定義模糊的用詞，如「一切相關」、「令其完全滿意」，可以被對方無限解釋，極易導致「範圍蠕變」(Scope Creep)。這會使我方被迫投入遠超預期的資源與人力，最終導致專案虧損。"
    },
    {
        "risk_level": "Medium",
        "clause_type": "不對等的責任上限 (Unequal or Low Limitation of Liability)",
        "keywords": [
            "limitation of liability", "liability cap", "consequential damages", "indirect damages", "disclaim",
            "責任上限", "賠償上限", "間接損害", "衍生性損害", "免責"
        ],
        "risk_analysis": "責任上限是保護雙方的條款，但若上限設定過低（例如，僅一個月的服務費）或雙方不對等，則可能在我方因對方重大違約而蒙受巨大損失時，無法獲得足夠的補償，風險與回報不成正比。"
    },
    {
        "risk_level": "Medium",
        "clause_type": "過於寬泛的禁止招攬 (Overly Broad Non-Solicitation)",
        "keywords": [
            "non-solicitation", "solicit", "poach", "employee", "contractor", "former employee", "indirectly",
            "禁止招攬", "挖角", "員工", "承包商", "前員工", "直接或間接"
        ],
        "risk_analysis": "範圍過於寬泛（例如，包含對方所有員工，而非僅限於接觸過本專案的人員）的禁止招攬條款，可能導致我方在公開市場上無法雇用優秀人才，不合理地限制了公司的人才引進策略。"
    },
    {
        "risk_level": "Medium",
        "clause_type": "合約的自動續約 (Automatic Renewal)",
        "keywords": [
            "automatic renewal", "automatically renew", "term", "notice of non-renewal",
            "自動續約", "合約期限", "不續約通知"
        ],
        "risk_analysis": "此條款存在操作風險。若因行政疏忽而忘記在指定期限內發出不續約通知，公司將被迫續約，即使該合約已不符合當前的商業利益或有更佳的替代方案，造成不必要的財務承擔。"
    },
    {
        "risk_level": "Medium",
        "clause_type": "寬鬆的轉讓權利 (Permissive Assignment Clause)",
        "keywords": [
            "assignment", "assign", "transfer", "without prior written consent", "change of control",
            "轉讓", "讓與", "移轉", "未經事先書面同意", "控制權變更"
        ],
        "risk_analysis": "如果條款允許對方在未經我方同意的情況下，隨意將合約轉讓給第三方（甚至可能是我們的競爭對手或財務狀況不佳的公司），這將給我方帶來巨大的不確定性與潛在的合作風險。"
    },

    # =================================================================
    # ==               🔍 需注意條款 (NOTICEABLE CLAUSE)              ==
    # ==   *風險較低，但應根據具體商業情境進行確認或微調* ==
    # =================================================================
    {
        "risk_level": "Low", # Pydantic model might need adjustment for "NOTICEABLE"
        "clause_type": "標準管轄法律與法院 (Standard Governing Law & Jurisdiction)",
        "keywords": [
            "governing law", "jurisdiction", "venue", "applicable law", "choice of law",
            "管轄法律", "準據法", "管轄法院", "適用法律"
        ],
        "risk_analysis": "這是標準條款，風險較低。但需注意確保選擇的法律與法院對我方是中立且方便的。若被指定在一個遙遠且不熟悉的司法管轄區，將會大幅增加未來解決爭議的成本與難度。"
    },
    {
        "risk_level": "Low",
        "clause_type": "合理的保密期限 (Reasonable Confidentiality Period)",
        "keywords": [
            "confidentiality", "non-disclosure", "NDA", "confidential information", "term",
            "保密", "機密資訊", "保密協議", "期限"
        ],
        "risk_analysis": "風險較低。一般商業秘密的保密期為3到5年是市場標準。但需注意，若涉及公司的核心技術秘密或永久性的商業機密，應爭取更長的保護期限，或要求該保密義務永久有效。"
    },
    {
        "risk_level": "Low",
        "clause_type": "不可抗力條款 (Force Majeure)",
        "keywords": [
            "force majeure", "act of god", "natural disaster", "pandemic", "epidemic", "cyber attack",
            "不可抗力", "天災", "自然災害", "疫情", "網路攻擊"
        ],
        "risk_analysis": "風險較低，為標準免責條款。但需審視其定義的事件範圍是否公平且與時俱進。例如，在今日的商業環境中，是否明確包含『疫情』、『政府行為』與『大規模網路攻擊』等非傳統天災事件，對雙方都很重要。"
    },
    {
        "risk_level": "Low",
        "clause_type": "通知條款 (Notices)",
        "keywords": [
            "notices", "written notice", "email", "address", "contact person",
            "通知", "書面通知", "電子郵件", "地址", "聯絡人"
        ],
        "risk_analysis": "風險極低，為行政管理條款。但審閱時應確保我方指定的收件地址、聯絡人與電子郵箱是正確且會被監控的，以避免錯過任何重要的法律通知（如違約通知、終止通知），從而導致權利喪失。"
    },
]

def get_risk_rubric_string():
    """將知識庫格式化為一個清晰的字串，方便注入到 Prompt 中。"""
    rubric_parts = []
    for item in CONTRACT_RISK_KNOWLEDGE_BASE:
        part = (
            f"- **風險等級 (Risk Level): {item['risk_level']}**\n"
            f"  - **條款類型 (Clause Type)**: {item['clause_type']}\n"
            f"  - **常見關鍵字 (Keywords)**: {', '.join(item.get('keywords', ['N/A']))}\n"
            f"  - **商業風險分析 (Business Risk Analysis)**: {item['risk_analysis']}\n"
        )
        rubric_parts.append(part)
    return "\n".join(rubric_parts)
