# %%
import docker

import paramiko
import os


class MySFTPClient(paramiko.SFTPClient):
    def put_dir(self, source, target):
        """Uploads the contents of the source directory to the target path. The
        target directory needs to exists. All subdirectories in source are
        created under target.
        """
        for item in os.listdir(source):
            if os.path.isfile(os.path.join(source, item)):
                self.put(os.path.join(source, item), "%s/%s" % (target, item))
            else:
                self.mkdir("%s/%s" % (target, item), ignore_existing=True)
                self.put_dir(os.path.join(source, item), "%s/%s" % (target, item))

    def mkdir(self, path, mode=511, ignore_existing=False):
        """Augments mkdir by adding an option to not fail if the folder exists"""
        try:
            super(MySFTPClient, self).mkdir(path, mode)
        except IOError:
            if ignore_existing:
                pass
            else:
                raise


client = paramiko.client.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

client.connect(
    "20.29.32.26",
    22,
    "azureuser",
)  # , password=PASSWORD)
transport = client.get_transport()
sftp = MySFTPClient.from_transport(transport)
sftp.mkdir("/tmp/test/", ignore_existing=True)
sftp.put_dir("test_data/datasets/battledim", "/tmp/test")
sftp.close()


# client = docker.from_env()
# # client.images.search("ghcr.io/ldimbenchmark/dualmethod:0.1.21")
# test = client.images.get("ghcr.io/ldimbenchmark/dualmethod:0.1.21")
# print("log")


# client = docker.DockerClient(base_url="ssh://azureuser@20.29.32.26")
# print(client.containers.list())


# %%
