import requests
import base64
import re
import os
import json
import argparse
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
from xml.etree import ElementTree

# check that we have a valid token in the env
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable is not set")

headers = {
    'Authorization': f'token {GITHUB_TOKEN}',
    "Accept": "application/vnd.github.v3+json"
}

file_patterns = ["pom.xml", "build.gradle"]

# Fetch the organizations
def fetch_organizations():
    # GitHub API base URL for getting orgs
    url = f"https://api.github.com/user/orgs"
    
    organizations = []
    page = 1
    
    while True:
        response = requests.get(f"{url}?page={page}", headers=headers)
        response_data = response.json()

        # If there are no organizations, exit the loop
        if not response_data:
            break

        organizations.extend(response_data)

        # Check if there is another page of organizations
        if 'next' not in response.links:
            break
        
        page += 1
    
    return organizations

def search_repositories(org_name, page=1):
    url = f'https://api.github.com/search/repositories?q=org:{org_name}+language:Java&page={page}'
    response = requests.get(url, headers=headers)
    return response.json()

def get_file_content(owner, repo, file_path):
    url = f'https://api.github.com/repos/{owner}/{repo}/contents/{file_path}'
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = response.json()['content']
        return base64.b64decode(content).decode('utf-8')
    return None

def search_files_in_repo(repo_owner, repo_name, file_patterns):
    # GitHub search API URL for searching code
    url = f"https://api.github.com/search/code"

    # Build the query string to search for build.gradle or pom.xml files
    query = " OR ".join([f"filename:{file_pattern}" for file_pattern in file_patterns])

    # List to hold the found file paths
    found_files = []

    # Pagination handling
    page = 1
    while True:
        # Send the search request
        params = {
            "q": query,
            "repo": f"{repo_owner}/{repo_name}",
            "page": page,
            "per_page": 100  # Adjust per_page as needed (max 100)
        }

        response = requests.get(url, headers=headers, params=params)
        
        # Check if the request was successful
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break

        # Parse the response
        data = response.json()

        # Collect the file information from the results
        for item in data.get('items', []):
            file_path = item['path']
            found_files.append(file_path)

        # Check if there is another page of results
        if 'next' in response.links:
            page += 1
        else:
            break

    return found_files

def parse_gradle_dependencies(file_content):
    dependencies = []

    # Updated regex to match dependencies in various formats (without lookbehind)
    regex = r'\b(implementation|api|compile|runtimeOnly|testImplementation|testCompile)\s+([\'"])([^:\']+):([^:\']+):([^\'"]+)\2'

    # Find all matches
    matches = re.findall(regex, file_content)

    # Add matches to the dependencies set
    for match in matches:
        # The match will contain the type (e.g., 'implementation') and the three parts of the dependency (group, artifact, version)
        if "com.thehutgroup" in match[2] or "com.thg" in match[2]:
            dependencies.append(f"{match[2]}:{match[3]}:{match[4]}")
    
    return dependencies

def parse_dependencies(file_content, file_type):
    dependencies = []
    
    if file_type == 'pom.xml':
        root = ElementTree.fromstring(file_content)
        namespaces = {'': 'http://maven.apache.org/POM/4.0.0'}
        for dependency in root.findall('.//dependency', namespaces):
            group_id = dependency.find('groupId', namespaces).text if dependency.find('groupId', namespaces) is not None else None
            if "com.thehutgroup" in group_id or "com.thg" in group_id:
                artifact_id = dependency.find('artifactId', namespaces).text if dependency.find('artifactId', namespaces) is not None else None
                version = dependency.find('version', namespaces).text if dependency.find('version', namespaces) is not None else None
                dependencies.append(f"{group_id}:{artifact_id}:{version}")
    
    elif file_type == 'build.gradle':
        dependencies = parse_gradle_dependencies(file_content)

    return dependencies

def fetch_all_repositories(org_name):
    all_repos = []
    page = 1
    
    while True:
        response_data = search_repositories(org_name, page)
        repos = response_data.get('items', [])
        
        if not repos:
            break
        
        all_repos.extend(repos)
        
        # Check for the 'next' page in the Link header
        if 'next' not in response_data.get('links', {}):
            break
        
        page += 1  # Move to the next page
        
    return all_repos

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    return data

def extract_service_deps(data):
    service_deps = {}

    for group, services in data.items():
        if isinstance(services, dict):
            for service, dependencies in services.items():
                service_deps[service] = set(extract_dep_name(dep) for dep in dependencies)

    return service_deps


def extract_dep_name(dep):
    # Extract the dependency name without the version
    # The dependency format is "group:artifact:version", so split by ':' and take the first two parts
    parts = dep.split(':')
    return ':'.join(parts[:2]) # ignore the version number

def histogram_counts(data):
    dep_count = Counter()

    # traverse data set to count the occurences of each dependency
    for group, services in data.items():
        if isinstance(services, dict):
            for service, deps in services.items():
                for d in deps:
                    dep_name = extract_dep_name(d)
                    dep_count[dep_name] += 1

    return dep_count

def plot_histogram(dep_count):
    labels, counts = zip(*dep_count.items())

    fig = go.Figure([go.Bar(x=labels, y=counts)])
    fig.update_layout(title="Histogram of Dependency Counts",
                        xaxis_title="Dependency",
                        yaxis_title="Count")
    fig.show()

def apply_count_clustering(dep_count, n_clusters=3):
    df = pd.DataFrame(dep_count.items(), columns=["Dependency", "Count"])

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters)
    df['Cluster'] = kmeans.fit_predict(df[['Count']])

    return df, kmeans


def apply_service_clustering(service_deps, n_clusters=3):
    mlb = MultiLabelBinarizer()

    dep_matrix = mlb.fit_transform(service_deps.values())
    df = pd.DataFrame(dep_matrix, columns=mlb.classes_, index=service_deps.keys())

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df)

    return df, kmeans

def plot_count_clusters(df):
    fig = px.scatter(df, x="Dependency", y="Count", color="Cluster", title="Dependency Clusters",
                     labels={"Dependency": "Dependency", "Count": "Count of Occurrences"})
    fig.update_traces(marker=dict(size=12))
    fig.show()

def plot_service_clusters(df):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df.drop(columns=['Cluster']))

    # Add the PCA components and the cluster labels to the DataFrame
    df['PCA1'] = pca_components[:, 0]
    df['PCA2'] = pca_components[:, 1]

    # Plot using Plotly
    fig = px.scatter(df, x="PCA1", y="PCA2", color="Cluster", text=df.index,
                     title="Service Clusters Based on Dependencies",
                     labels={"PCA1": "Principal Component 1", "PCA2": "Principal Component 2"})
    fig.update_traces(marker=dict(size=12), textposition="top center")
    fig.show()

def main(filename=None, plot='histogram'):

    if filename:
        data = load_json(filename)
        dep_count = histogram_counts(data)
        service_deps = extract_service_deps(data)

        if plot == 'histogram':
            plot_histogram(dep_count)

        if plot == 'count-scatter':
            df, kmeans = apply_count_clustering(dep_count)
            plot_count_clusters(df)

        if plot == 'service-scatter':
            df, kmeans = apply_service_clustering(service_deps)
            plot_service_clusters(df)

    else:
        print("fetching..")
        out = {}
        orgs = fetch_organizations()
        print(f"found {len(orgs)} organizations")
        for org in orgs:
            print(f"Org: {org['login']}")
            repos = fetch_all_repositories(org['login'])

            # new dict for this org
            org_d = {}

            for repo in repos:
                repo_name = repo['name']
                #print(f"\t- Checking repository {repo_name}")

                # Searching for build files in subdirs is too costly with respect to the github API
                # build_files = search_files_in_repo(org['login'], repo_name, file_patterns)

                # if build_files:
                #     print(f"found {len(build_files)} in subdirs:")

                #     for f in build_files:
                #         print(f"subdir build_file: {f}")

                #         # fetch the 
                #         if "pom.xml" in f:
                #             pom_content = get_file_content(repo['owner']['login'], repo_name, f)

                #         if "build.gradle" in f:
                #             gradle_content = get_file_content(repo['owner']['login'], repo_name, f)

                #         if pom_content:
                #             dependencies = parse_dependencies(pom_content, 'pom.xml')
                #             if len(dependencies) > 0:
                #                 print(f"Dependencies in {repo_name}: {dependencies}")
                        
                #         if gradle_content:
                #             dependencies = parse_dependencies(gradle_content, 'build.gradle')
                #             if len(dependencies) > 0:
                #                 print(f"Dependencies in {repo_name}: {dependencies}")

                # Try fetching root level pom.xml and build.gradle files
                pom_content = get_file_content(repo['owner']['login'], repo_name, 'pom.xml')
                gradle_content = get_file_content(repo['owner']['login'], repo_name, 'build.gradle')

                if pom_content:
                    dependencies = parse_dependencies(pom_content, 'pom.xml')
                    if len(dependencies) > 0:
                        #print(f"\t - pom dependencies in {repo_name}: {dependencies}")
                        org_d[repo_name] = dependencies

                elif gradle_content:
                    dependencies = parse_dependencies(gradle_content, 'build.gradle')
                    if len(dependencies) > 0:
                        #print(f"\t - gradle dependencies in {repo_name}: {dependencies}")
                        org_d[repo_name] = dependencies
                else:
                    #print(f"\t - No dependency file found in {repo_name}")
                    org_d[repo_name] = []

            # add the org info to the output
            out[org['login']] = org_d

        # Write dictionary to a JSON file
        with open('output.json', 'w') as json_file:
            json.dump(out, json_file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="dependency file in json format")
    parser.add_argument('--dep_file', type=str, help="path to dependency file", default=None);
    parser.add_argument('--plot', type=str, help="Type of plot: histogram|scatter")

    args = parser.parse_args()
    main(args.dep_file, args.plot)