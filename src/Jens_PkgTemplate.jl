using PkgTemplates

t=Template(;
           user="Cywtim",
           dir="/Users/cyan/Documents/VScodeProjects/Jens",
           authors="Cywtim",
           julia=v"1.10.1",
           plugins=[
               License(; name="MIT"),
               Git(; manifest=true),
               GitHubActions(; x86=true),
               Documenter{GitHubActions}(),
               Codecov(),
               Develop(),
               TravisCI(),
               Coveralls(),
           ],
       )

t("Jens")