How do I run this?
==================

Clone it somewhere. Get into the root of the project (the folder containing this file) and follow through.

First things first, instructions usually start with installation. And who doesn't like to install some Python dependencies?

The only way I know is through [pip][pip]:

```bash
$ pip install -r requirements.txt
```

Now crawl them speeches, man!

```bash
$ scrapy crawl discursos
```

> NOTE: If it fails because of [MongoDB][mongodb], make sure you have a working [MongoDB][mongodb] server running on your machine. Or edit `settings.py` and change the connection settings.

Now this is gonna take some time to finish. It'll crawl about to 95k speeches (last time I did it), so yeah, it's going to take some time. I suggest looking at [cat pictures on Reddit](http://www.reddit.com/r/catpictures).

When it's done, it's done.

If you got too many errors or you scraped files are missing their files (this is, the content of the speech), you should try running the retry command:

```bash
$ scrapy crawl teordiscursos
```

You'll probably notice that the remote web service uses some sort of rate limiting powered by `503`s. This little command may come in hand:

```bash
$ watch -n120 'scrapy crawl teordiscursos -a limit=2000 --loglevel=INFO | egrep "(file_status_count|download\/)"'
```

[pip]: https://pypi.python.org/pypi/pip
[scrapy]: http://scrapy.org/
[mongodb]: https://www.mongodb.org/
