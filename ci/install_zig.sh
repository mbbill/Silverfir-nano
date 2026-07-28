#!/usr/bin/env bash
# Pinned Zig installer for GitHub Actions cross-target jobs.
set -euo pipefail

version="0.14.0"
archive="zig-linux-x86_64-${version}.tar.xz"
url="https://ziglang.org/download/${version}/${archive}"
sha256="473ec26806133cf4d1918caf1a410f8403a13d979726a9045b421b685031a982"
install_dir="${RUNNER_TOOL_CACHE:?RUNNER_TOOL_CACHE is required}/zig/${version}/x86_64"

if [[ ! -x "${install_dir}/zig" ]]; then
    temp_dir="$(mktemp -d)"
    trap 'rm -rf -- "${temp_dir}"' EXIT

    curl --fail --location --retry 3 --output "${temp_dir}/${archive}" "${url}"
    printf '%s  %s\n' "${sha256}" "${temp_dir}/${archive}" | sha256sum --check -
    tar --extract --xz --file "${temp_dir}/${archive}" --directory "${temp_dir}"

    mkdir -p "$(dirname "${install_dir}")"
    mv "${temp_dir}/zig-linux-x86_64-${version}" "${install_dir}"
fi

echo "${install_dir}" >> "${GITHUB_PATH:?GITHUB_PATH is required}"
"${install_dir}/zig" version
