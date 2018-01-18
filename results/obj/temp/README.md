This folder is used by data.py to temporary save data files while the agent still collects data. The purpose of this is obviously to gain performance, by saving batches and don't let the memory consumption grow very much.

If this folder does not exists, data.Data.temp_save() will crash.
