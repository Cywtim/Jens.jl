using PkgTemplates

t=Template(;
           user="Cywtim",
           dir="/Users/cyan/Documents/VScodeProjects/Jens-private/Package/",
           authors="Cywtim",
           julia=v"1.10.1",
           plugins=[
               License(; name="MIT"),
               Git(; manifest=true),
               GitHubActions(; x86=true),
               Documenter{GitHubActions}(),
               Codecov(),
               Develop(),
               GitLabCI(),
               Coveralls(),
           ],
       )

t("Jens")