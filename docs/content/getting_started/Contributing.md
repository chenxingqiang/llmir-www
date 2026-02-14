---
title: "How to Contribute"
date: 2019-11-29T15:26:15Z
lastmod: 2025-02-04T15:26:15Z
draft: false
weight: 15
---

Everyone is welcome to contribute to LLMIR. The project uses the [llmir GitHub repository](https://github.com/chenxingqiang/llmir) for code and issues.

There are several ways of getting
involved and contributing including reporting bugs, improving documentation and
tutorials.

## Community Guidelines

Please be mindful of the [LLVM Code of Conduct](https://llvm.org/docs/CodeOfConduct.html),
which pledges to foster an open and welcoming environment.

### Contributing code

Please send [pull requests](https://github.com/chenxingqiang/llmir/pulls) on GitHub. If you don't have write access, fork the repo and open a PR from your fork.

#### Commit messages

Follow the git conventions for writing a commit message, in particular the
first line is the short title of the commit. The title should be followed by an
empty line and a longer description. Prefer describing *why* the change is
implemented rather than what it does. The latter can be inferred from the code.
This [post](https://chris.beams.io/posts/git-commit/) give examples and more
details.

### Issue tracking

To report a bug or request a feature, use the [LLMIR GitHub Issues](https://github.com/chenxingqiang/llmir/issues).

If you want to contribute, browse the [repository](https://github.com/chenxingqiang/llmir) and [open issues](https://github.com/chenxingqiang/llmir/issues). Comment on an issue to indicate you're working on it.

### Contribution guidelines and standards

*   Read the [Developer Guide](DeveloperGuide.md).
*   Ensure that you use the correct license. Examples are provided below.
*   Include tests when you contribute new features, as they help to a)
    prove that your code works correctly, and b) guard against future breaking
    changes to lower the maintenance cost.
*   Bug fixes also generally require tests, because the presence of bugs
    usually indicates insufficient test coverage.

#### License

Include a license at the top of new files. LLMIR uses Apache-2.0 with LLVM exceptions. See [LICENSE.TXT](https://github.com/chenxingqiang/llmir/blob/main/LICENSE.TXT) in the repository.
