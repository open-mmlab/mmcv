## 拉取请求

### 什么是拉取请求？
`拉取请求` (Pull Request), [GitHub 官方文档](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)定义如下。

>拉取请求是一种通知机制。你修改了他人的代码，将你的修改通知原来作者，希望他合并你的修改。

### 基本的工作流，如下：
1. 复刻并拉取最新的 OpenMMLab 代码库
2. 新建并检出（checkout）一个新的分支进行开发
3. 提交修改
4. 创建一个`拉取请求`

### 具体步骤
1. 复刻 OpenMMLab 原代码库，点击 GitHub 页面右上角的 **Fork** 按钮即可。 \
![avatar](../_static/community/1.png)
2. 克隆复刻的代码库到本地
```bash
git clone git@github.com:XXX/mmcv.git
```
3. 添加原代码库为上游代码库
```bash
git remote add upstream git@github.com:open-mmlab/mmcv
```
4. 拉取最新的原代码库的主分支
```bash
git pull upstream master
```
5. 新建并检出一个新的分支，进行开发
```bash
git checkout -b branchname
# coding
git add [files]
git commit -m 'messages'
```
6. 推送到复刻的代码库
```bash
git push origin branchname
```
7. 创建一个`拉取请求`
![avatar](../_static/community/2.png)
8. 修改`拉取请求`信息模板，描述修改原因和修改内容。
9. 创建`拉取请求`时，可以关联给相关人员进行 review
![avatar](../_static/community/3.png)
10. 关联相关的`议题` (issue) 和`拉取请求`
11. 根据 reviewer 的意见修改代码，并推送修改
12. `拉取请求`合并之后删除该分支
```bash
git branch -d branchname # delete local branch
git push origin --delete branchname # delete remote branch
```
### PR 规范
1. 使用 [pre-commit hook](https://pre-commit.com)，尽量减少代码风格相关问题
2. 一个PR对应一个短期分支
3. 粒度要细，一个PR只做一件事情，避免超大的PR
>- Bad:实现Faster R-CNN
>- Acceptable:给 Faster R-CNN 添加一个 box head
>- Good:给 box head 增加一个参数来支持自定义的 conv 层数
4. 每次 Commit 时需要提供清晰且有意义 commit 信息
5. 提供清晰且有意义的`拉取请求`描述
>- 标题写明白任务名称，一般格式:[Prefix] Short description of the pull request (Suffix)
>- prefix: 新增功能 [Feature], 修 bug [Fix], 文档相关 [Docs], 开发中 [WIP] (暂时不会被review)
>- 描述里介绍`拉取请求`的主要修改内容，结果，以及对其他部分的影响, 参考`拉取请求`模板
>- 关联相关的`议题` (issue) 和其他`拉取请求`