FROM wolfsoftwareltd/pyenv-amazonlinux

RUN yum -y update && \
    yum -y groupinstall "Development Tools" && \
    yum -y install openssl-devel bzip2-devel libffi-devel && \
    yum -y install wget;


ENTRYPOINT ["tail"]
CMD ["-f","/dev/null"]