"""v2.18.1:存量数据卫生修复(只追加、可追溯)。

v2.18.0 迁移把旧 run 的证据导入时,``run_dir`` 缺失的 run 拿不到任务指纹,300+ 个
trial 与 2 个指纹成了 ``atid=None`` / ``unknown-unverified``:它们不参与同任务检索,
被关联到原则后却把原则的 ``abstract_task_ids`` 拖进"底座未知"档。本包按 run 目录
重新推出指纹,以**追加新行**的方式修正(run 级 trial 行取最新、指纹别名可撤销),
原始行不动;每次修复都写一条带证据的提交与变更日志。

- :mod:`.legacy_fingerprint`:`library fix-fingerprints`。
"""
