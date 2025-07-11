import nox

@nox.session
def tests(session):
    session.install("-e", ".", "pytest", "hypothesis", "coverage")
    session.run("coverage", "run", "-m", "pytest")
    session.run("coverage", "report", "-m")

@nox.session
def lint(session):
    session.install("ruff")
    session.run("ruff", "pointing", "scripts", "tests")
