# Lexsi Labs Source Available License (LSAL) – Version 1.1 (AlignTune)

## Preamble

This Source Available License governs use of the software known as **AlignTune**, together with its configuration files, recipes, notebooks, examples, and documentation (collectively, the "Licensed Work"), developed and owned by **Lexsi Labs (Lithasa Technologies Pvt. Ltd.)** ("Licensor").

This is **not** an open-source license as defined by the [Open Source Initiative (OSI)](https://opensource.org/). It grants broad, free access to the source code and artifacts for **research, evaluation, education, and audit**, while restricting commercial exploitation and unsafe use. These restrictions are deliberate: the Licensed Work performs post-training (supervised fine-tuning, preference optimization, reinforcement learning, distillation, and model merging) of language models, including workflows that can materially change a model's behavior and safety characteristics. Its value lies in enabling reproducible, auditable, and responsible post-training workflows, not in unlicensed commercial exploitation or the unreviewed removal of model safeguards.

---

## 1. Grant of Rights

Subject to the terms of this License, permission is hereby granted, free of charge, to any person obtaining a copy of the Licensed Work, to use, copy, modify, merge, publish, and redistribute the Licensed Work and derivative works thereof, **for Noncommercial Purposes only**, provided that the above copyright notice, this License, and the Responsible Use conditions (Section 4) are included in full in all copies or substantial portions of the Licensed Work.

**Noncommercial Purposes** means:

* personal use for research, experimentation, private study, or hobby projects;
* academic and scholarly research, teaching, and publication (including use in papers, theses, benchmarks, and reproducibility artifacts);
* internal evaluation, red-teaming, or safety auditing that is not itself a paid product or service;
* use by charitable organizations, educational institutions, public research organizations, or government bodies for non-revenue-generating purposes.

## 2. Commercial Restriction

Without a separate **commercial license** from Lexsi Labs, you may **not** Sell the Licensed Work.

"**Sell**" means practicing any right granted to you under this License to provide to third parties, for a fee or other consideration, a product or service whose value derives, entirely or substantially, from the functionality of the Licensed Work — including without limitation:

* offering the Licensed Work, or a derivative of it, as a commercial product, paid service, SaaS, hosted, or API offering;
* embedding the Licensed Work in proprietary or revenue-generating software;
* paid consulting or support whose substance is the Licensed Work.

You may also not **re-license, rebrand, or redistribute** the Licensed Work under different terms, nor use **Lexsi Labs**, **AlignTune**, or related trademarks, logos, or branding except to identify unmodified, licensed copies.

## 3. Patents

The Licensor grants you a non-exclusive, worldwide, royalty-free patent license, under patent claims the Licensor can license that are necessarily infringed by the Licensed Work, to use the Licensed Work for Noncommercial Purposes as permitted by this License.

This patent license terminates immediately if you or your company make any written claim that the Licensed Work infringes a patent.

## 4. Responsible Use (Safety Conditions)

The Licensed Work can fine-tune, align, post-train, distill, and merge language models, and can materially change a model's safety-relevant behavior (including its refusal, honesty, and harm-avoidance characteristics). As a condition of this License, you may **not**:

* use the Licensed Work to fine-tune, post-train, merge, or otherwise modify a model for the purpose of removing, weakening, or bypassing its safety behaviors or guardrails, except in a bona fide research, evaluation, red-teaming, or safety-auditing context subject to appropriate safeguards;
* use the Licensed Work to intentionally train, curate rewards for, or otherwise produce a model whose purpose or reasonably foreseeable use is to cause, elicit, or materially enable unlawful or materially harmful behavior, including behavior that promotes or facilitates violence, hatred, discrimination, targeted harassment, exploitation, fraud, malicious cyber activity, or other serious wrongdoing;
* knowingly deploy, distribute, publish, make available, or otherwise operationalize a model post-trained using the Licensed Work without re-evaluating its safety-relevant behavior after training, where that model is intended for production, user-facing, or agentic use.

## 5. Ownership

All rights, title, and interest in and to the Licensed Work remain with **Lithasa Technologies Pvt. Ltd.** Except as expressly stated in Sections 1 and 3, nothing in this License transfers ownership or any other rights to the Licensee.

## 6. Contributions

If you submit modifications, pull requests, or patches ("Contributions") to Lexsi Labs, you grant Lexsi Labs a perpetual, worldwide, royalty-free right to use, modify, distribute, and license your Contributions under any terms, including commercial ones, and you represent that you have the right to make such Contributions.

## 7. Warranty Disclaimer

THE LICENSED WORK IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE LICENSOR OR CONTRIBUTORS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE LICENSED WORK OR THE USE OR OTHER DEALINGS IN THE LICENSED WORK.

## 8. Termination

This License terminates automatically if you breach any of its terms. Upon termination, you must immediately cease use and destroy all copies of the Licensed Work in your possession. Licenses of parties who received the Licensed Work from you remain in force provided they remain in compliance.

## 9. Governing Law

This License shall be governed by and construed in accordance with the **laws of India**, without regard to its conflict of law principles.

## 10. Contact for Commercial Licensing

For commercial use, partnership, or redistribution rights, contact:
**support@lexsi.ai** · **https://lexsi.ai**

## 11. Notice

**AlignTune** © 2026 **Lithasa Technologies Pvt. Ltd.**
Licensed under the **Lexsi Labs Source Available License (LSAL) v1.1**.
**Not for commercial use, and not for removing or weakening the safety behaviors of a model, without explicit permission.**

---

## Third-Party Components

AlignTune vendors a patched copy of [mergekit](https://github.com/arcee-ai/mergekit) under `third_party/mergekit/`, which remains licensed under its own terms (GNU LGPL v3.0), and depends at install/runtime on other third-party packages (some under noncommercial or copyleft terms of their own). See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for the full list.
