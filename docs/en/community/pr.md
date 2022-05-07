## Pull Request (PR)

### What is PR

`PR` is the abbreviation of `Pull Request`. Here's the definition of `PR` in the [official document](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) of Github.

```
Pull requests let you tell others about changes you have pushed to a branch in a repository on GitHub. Once a pull request is opened, you can discuss and review the potential changes with collaborators and add follow-up commits before your changes are merged into the base branch.
```

### Basic Workflow

1. Get the most recent codebase
2. Checkout a new branch from the master branch
3. Commit your changes
4. Push your changes and create a PR
5. Discuss and review your code
6. Merge your branch to the master branch

### Procedures in detail

#### 1. Get the most recent codebase

- When you work on your first PR

  Fork the OpenMMLab repository: click the **fork** button at the top right corner of Github page
  ![avatar](../_static/community/1.png)

  Clone forked repository to local

  ```bash
  git clone git@github.com:XXX/mmcv.git
  ```

  Add source repository to upstream

  ```bash
  git remote add upstream git@github.com:open-mmlab/mmcv
  ```

- After your first PR

  Checkout master branch of the local repository and pull the latest master branch of the source repository

  ```bash
  git checkout master
  git pull upstream master
  ```

#### 2. Checkout a new branch from the master branch

```bash
git checkout -b branchname
```

```{tip}
To make commit history clear, we strongly recommend you checkout the master branch before create a new branch.
```

#### 3. Commit your changes

```bash
# coding
git add [files]
git commit -m 'messages'
```

#### 4. Push your changes to the forked repository and create a PR

- Push the branch to your forked remote repository

  ```bash
  git push origin branchname
  ```

- Create a PR
  ![avatar](../_static/community/2.png)

- Revise PR message template to describe your motivation and modifications made in this PR. You can also link the related issue to the PR manually in the PR message (For more information, checkout the [official guidance](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue)).

#### 5. Discuss and review your code

- After creating a pull request, you can ask a specific person to review the changes you've proposed
  ![avatar](../_static/community/3.png)

- Modify your codes according to reviewers' suggestions and then push your changes

#### 6.  Merge your branch to the master branch and delete the branch

```bash
git branch -d branchname # delete local branch
git push origin --delete branchname # delete remote branch
```

### PR Specs

1. Use [pre-commit](https://pre-commit.com) hook to avoid issues of code style

2. One short-time branch should be matched with only one PR

3. Accomplish a detailed change in one PR. Avoid large PR

   - Bad: Support Faster R-CNN
   - Acceptable: Add a box head to Faster R-CNN
   - Good: Add a parameter to box head to support custom conv-layer number

4. Provide clear and significant commit message

5. Provide clear and meaningful PR description

   - Task name should be clarified in title. The general format is: \[Prefix\] Short description of the PR (Suffix)
   - Prefix: add new feature \[Feature\], fix bug \[Fix\], related to documents \[Docs\], in developing \[WIP\] (which will not be reviewed temporarily)
   - Introduce main changes, results and influences on other modules in short description
   - Associate related issues and pull requests with a milestone
