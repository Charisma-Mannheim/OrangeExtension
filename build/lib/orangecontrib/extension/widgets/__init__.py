# Category description for the widget registry
import sysconfig



NAME = "Charisma toolbox"
DESCRIPTION = "Contains widgets to perform and evaluate ordinary linear regression and partial least squares regression on data," \
              "as well as widgets to perform and evaluate principal component analysis on data and enables linear discriminant analysis for raw data and score data from pca."
BACKGROUND = "#4AE714"
ICON = "icons/SVGlogo.svg"
PRIORITY = 7


# Location of widget help files.
WIDGET_HELP_PATH = (
    # Development documentation
    # You need to build help pages manually using
    # make htmlhelp
    # inside doc folder
    #
    ("{DEVELOP_ROOT}/doc/_build/html/index.html", None),

    # Documentation included in wheel
    # Correct DATA_FILES entry is needed in setup.py and documentation has to be built
    # before the wheel is created.
    #orange
    ("{}/help/orange3-PCAExtension/index.html".format(sysconfig.get_path("data")), None),

    # Online documentation url, used when the local documentation is not available.
    # Url should point to a page with a section Widgets. This section should
    # includes links to documentation pages of each widget. Matching is
    # performed by comparing link caption to widget name.
    #orange
    ("http://orange3-PCAExtension-addon.readthedocs.io/en/latest/", "")
)
#print(Deve)