import os
import tempfile
import zipfile
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import requests
from requests.adapters import HTTPAdapter
from tqdm.notebook import tqdm
from urllib3.util.retry import Retry

from peaks.core.fileIO.data_loading import load
from peaks.core.utils.misc import analysis_warning

ROOT_URL = "https://zenodo.org/api/records/15928652/files"


class ZenodoDownloader:
    """Helper class to download files from Zenodo.

    If the data is not public, a Zenodo token must be provided. This will be taken
    from the environment variable `ZENODO_TOKEN` if not explicitly provided."""

    def __init__(self, file_list, token=None):
        self.root_url = ROOT_URL.rstrip("/")
        self.file_list = file_list
        self.token = token or os.getenv("ZENODO_TOKEN")
        self._tempdir_context = None
        self.downloaded_files = {}

    def _make_headers_if_needed(self, url):
        if "draft" in url and self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    def _download_with_progress(self, url, dest_path):
        headers = self._make_headers_if_needed(url)

        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        proxies = {"http": os.getenv("http_proxy"), "https": os.getenv("https_proxy")}

        try:
            response = session.get(
                url,
                headers=headers,
                proxies=proxies,
                stream=True,
                timeout=(10, 300),  # (connect timeout, read timeout)
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download {url}: {e}") from e

        total = int(response.headers.get("content-length", 0))

        with (
            open(dest_path, "wb") as f,
            tqdm(
                desc=os.path.basename(dest_path),
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=False,
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
                    f.flush()
                    bar.update(len(chunk))

    def download(self):
        if self._tempdir_context is not None:
            return self.downloaded_files

        self._tempdir_context = tempfile.TemporaryDirectory()
        tmpdir = self._tempdir_context.name

        with tqdm(
            total=len(self.file_list), desc="Downloading sample file(s)", unit="file"
        ) as overall_bar:
            for filename in self.file_list:
                url = f"{self.root_url}/{filename}/content"
                dest_path = os.path.join(tmpdir, filename)
                self._download_with_progress(url, dest_path)
                self.downloaded_files[filename] = dest_path
                overall_bar.update(1)

        return self.downloaded_files

    @property
    def fpath(self):
        """Path to the temporary directory containing downloaded files."""
        if not self.downloaded_files:
            raise ValueError(
                "No files have been downloaded yet or the temporary files "
                "have been deleted. Call `download()` to download the files."
            )
        return self._tempdir_context.name

    def cleanup(self, quiet=False):
        if self._tempdir_context is not None:
            self._tempdir_context.cleanup()
            self._tempdir_context = None
            self.downloaded_files = {}
            analysis_warning(
                "Temporary files have been deleted. You can now safely close the notebook.",
                title="Cleanup Complete",
                warn_type="success",
                quiet=quiet,
            )
        else:
            analysis_warning(
                "No temporary files to clean up. The cleanup method has already been "
                "called or no files were downloaded.",
                title="No Cleanup Needed",
                warn_type="info",
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup(quiet=True)


def get_tutorial1_data():
    """Download the tutorial 1 data from Zenodo."""
    file_list = ["i05-59818.nxs", "i05-59819.nxs", "i05-59853.nxs"]
    downloader = ZenodoDownloader(file_list)
    analysis_warning(
        "Downloading data files for the Tutorial to a temporary directory. The file path"
        " is available via the .fpath attribute. Make sure to call the `cleanup()` "
        "method at the end of the tutorial to delete the temporary files.",
        title="Sample Data Download",
        warn_type="warning",
    )
    downloader.download()

    return downloader


class ExampleData:
    """Class to access example data files for the Peaks package. The data is
    downloaded from Zenodo and cached for subsequent access.
    The files are available as class methods, e.g., `ExampleData.dispersion()`."""

    _cache = {}

    @classmethod
    def _get_and_load(cls, fname, **kwargs):
        """Load data from a file, downloading it if not already cached.
        kwargs are passed to the `load` function."""
        data = cls._cache.get(fname)
        if data is None:
            with ZenodoDownloader([fname]) as downloader:
                downloader.download()
                fpath = os.path.join(downloader.fpath, fname)
                data = load(fpath, **kwargs)
                cls._cache[fname] = data
        return data.copy(deep=True)  # Return a copy to avoid modifying the cached data

    @classmethod
    def _get_and_load_from_zip(cls, fname, **kwargs):
        """Load data from a file inside a zip archive, downloading it if not already
        cached. The zip file is downloaded, extracted, and the specified file is loaded.
        kwargs are passed to the `load` function."""
        data = cls._cache.get(fname)
        if data is None:
            with ZenodoDownloader([fname]) as downloader:
                downloader.download()
                fpath = os.path.join(downloader.fpath, fname)
                # unzip the file
                with zipfile.ZipFile(fpath, "r") as zip_ref:
                    zip_ref.extractall(downloader.fpath)
                # Load the data from the extracted file
                fpath = os.path.join(downloader.fpath, os.path.splitext(fname)[0])
                data = load(fpath, **kwargs)
                cls._cache[fname] = data
        return data.copy(deep=True)  # Return a copy to avoid modifying the cached data

    @classmethod
    def dispersion(cls):
        return cls._get_and_load("i05-59819.nxs")

    @classmethod
    def dispersion2a(cls):
        return cls._get_and_load("210326_GM2-667_GK_1.xy")

    @classmethod
    def dispersion2b(cls):
        return cls._get_and_load("210326_GM2-667_GK_2.xy")

    @classmethod
    def dispersion2c(cls):
        return cls._get_and_load("210326_GM2-667_GK_3.xy")

    @classmethod
    def dispersion3(cls):
        return cls._get_and_load("i05-1-34301.nxs")

    @classmethod
    def dispersion4(cls):
        return cls._get_and_load("i05-1-31473.nxs")

    @classmethod
    def gold_reference(cls):
        return cls._get_and_load("i05-59853.nxs")

    @classmethod
    def gold_reference2(cls):
        return cls._get_and_load("Gold.xy")

    @classmethod
    def gold_reference3(cls):
        return cls._get_and_load("i05-70214.nxs")

    @classmethod
    def gold_reference4(cls):
        return cls._get_and_load("Ep20eV.xy")

    @classmethod
    def FS(cls):
        return cls._get_and_load("i05-59818.nxs")

    @classmethod
    def hv_map(cls):
        return cls._get_and_load("i05-69294.nxs")

    @classmethod
    def SM(cls):
        return cls._get_and_load("i05-1-24270_sm.nc")

    @classmethod
    def nano_focus(cls):
        return cls._get_and_load("i05-1-49292.nxs")

    @classmethod
    def nano_focus_w_I0norm(cls):
        return cls._get_and_load("i05-1-49292.nxs", norm_by_I0=True)

    @classmethod
    def tr_arpes(cls):
        # Need to download and unzip the file to simulate the original data structure
        return cls._get_and_load_from_zip("029 Gr.zip")

    @classmethod
    def tr_arpes2(cls):
        return cls._get_and_load_from_zip("028 Gr.zip")

    @classmethod
    def xps(cls):
        return cls._get_and_load("i05-1-49260.nxs")

    @classmethod
    def structure(cls):
        data = cls._cache.get("structure")
        if data is None:
            url = "https://www.crystallography.net/cod/4515175.cif"
            response = requests.get(url)
            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(
                    "w+", suffix=".cif", delete=True
                ) as tmp:
                    tmp.write(response.text)
                    tmp.flush()
                    data = load(tmp.name)
                    cls._cache["structure"] = data
            else:
                raise ValueError(
                    f"Failed to download CIF. HTTP status code: {response.status_code}"
                )
        return deepcopy(data)

    @classmethod
    def cleanup(cls):
        cls._cache.clear()


def plot_tutorial_example_figure(fname, figsize=(16, 8)):
    """Plot an example figure in the tutorial notebooks."""
    tutorial_dir = os.path.join(
        Path(__file__).resolve().parent.parent.parent.parent, "tutorials", "figs"
    )
    fig_path = os.path.join(tutorial_dir, fname)
    plt.figure(figsize=figsize)
    plt.imshow(plt.imread(fig_path))
    plt.axis("off")
    plt.show()
