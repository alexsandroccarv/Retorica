How do I run this?
==================

Clone it somewhere. Get into the root of the project (the folder containing this file) and follow through.

First things first, instructions usually start with installation. And who doesn't like to install some Python dependencies?

The only way I know is through [pip][pip]:

```bash
$ pip install scrapy scrapy_mongodb xmltodict fake_useragent
```

Now crawl them speeches, man!

```bash
$ scrapy crawl discursos
```

> NOTE: If it fails because of [MongoDB][mongodb], make sure you have a working [MongoDB][mongodb] server running on your machine. Or edit `settings.py` and change the connection settings.

Now this is gonna take some time to finish. It'll crawl about to 95k speeches (last time I did it), so yeah, it's going to take some time. I suggest looking at [cat pictures on Reddit](http://www.reddit.com/r/catpictures).

When it's done, it's done. If you got too many errors you can try running the retry (`scrapy retrydownloads`) command, which may or may not work.

[pip]: https://pypi.python.org/pypi/pip
[scrapy]: http://scrapy.org/
[mongodb]: https://www.mongodb.org/
