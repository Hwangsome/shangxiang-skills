# 商详 Skills：裁剪 · 翻译 · 上传合规

三个给 AI（Claude）用的 Skill，用来批量处理电商商品详情图（商详图）：

| Skill | 做什么 | 你可以这样说 |
|---|---|---|
| [`shangxiang-crop`](skills/shangxiang-crop/SKILL.md) | 把商详长图按版块边界裁成完整的分图，不切断文字和照片 | “帮我把 商详 文件夹里的详情图裁剪成小图” |
| [`shangxiang-translate-ms`](skills/shangxiang-translate-ms/SKILL.md) | 把分图上的中文换成马来文，产品照片和排版不变 | “把裁好的分图翻译成马来文” |
| [`shangxiang-upload-check`](skills/shangxiang-upload-check/SKILL.md) | 检查并修复上传规则：宽 ≥700、高 ≤ 宽×2、≤2MB | “检查这些图能不能上传，不合规的修一下” |

三个 Skill 可以串起来用：**裁剪 → 翻译 → 合规检查 → 上架**。你只需要说一句：“把这批商详裁好、翻成马来文，再检查能不能上传。”

## Skill 是什么？

Skill 是写给 AI 的岗位操作手册。每个 Skill 是一个文件夹：

```
shangxiang-crop/
├── SKILL.md          ← 手册正文：什么时候用、怎么做、做到什么标准
├── requirements.txt  ← 需要的 Python 库
└── scripts/          ← 配套工具，AI 直接调用
```

AI 遇到对应的活，会自动翻开手册照着做，不用你每次从头交代。

## 安装（Claude Code）

1. 下载本仓库：点页面右上角绿色的 **Code → Download ZIP** 并解压。会用 git 的话也可以：

   ```bash
   git clone https://github.com/Hwangsome/shangxiang-skills.git
   ```

2. 把 `skills/` 里的三个文件夹复制到 Claude 的 Skill 目录：
   - Windows：`C:\Users\<你的用户名>\.claude\skills\`
   - macOS / Linux：`~/.claude/skills/`

3. 安装 Python 依赖（每个 Skill 文件夹里都有 `requirements.txt`）：

   ```bash
   pip install -r shangxiang-crop/requirements.txt
   pip install -r shangxiang-upload-check/requirements.txt
   pip install -r shangxiang-translate-ms/requirements.txt   # 翻译需要的库较多，建议有 NVIDIA 显卡
   ```

4. 重新打开 Claude Code，直接用平常话说需求就行。也可以输入 `/shangxiang-crop` 点名使用某个 Skill。

## 使用提醒

- **AI 会出错，人要验收。** 处理完一定抽查几张，发现问题就直接告诉 AI 哪里不对。
- **规则变了要更新手册。** 平台改了上传要求，或者换了目标市场，记得修改对应的 `SKILL.md`。
- 本仓库只包含脚本和说明，不包含任何商品图片或商品词表。翻译用的词表会在你第一次运行时，在你自己的工作目录里生成。

## 许可

[MIT](LICENSE)
